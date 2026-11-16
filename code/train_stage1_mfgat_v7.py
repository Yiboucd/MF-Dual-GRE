#!/usr/bin/env python3
"""Tune Stage-1 MFGAT with masked-anchor self-supervision.

Dense maps are used only by the existing validation/test evaluator. During
training, a random subset of sparse node-frequency anchors is removed from the
model input and reconstructed from the remaining anchors, geometry, and graph.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_stage1_controlled_v6 import FrequencyAwareMFGAT, ResidualGATBlock
from train_stage1_label_free_v5 import (
    MMWAVE,
    NF,
    PureGroupedMFGAT,
    SUB6,
    evaluate,
    frequency_balanced_loss,
    graph_batches,
    graph_smoothness,
    make_figure,
)
from train_clean_superpixel_gat_v2 import load_examples


class AnchorPreservingMFGAT(nn.Module):
    """Blend region predictions with anchors visible to the model."""

    def __init__(self, core: nn.Module, consistency_alpha: float = 1.0):
        super().__init__()
        self.core = core
        self.consistency_alpha = float(consistency_alpha)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raw = self.core(x, edge_index)
        observed = x[:, :NF]
        mask = x[:, NF : 2 * NF]
        alpha = torch.as_tensor(
            self.consistency_alpha,
            dtype=raw.dtype,
            device=raw.device,
        )
        return raw + alpha * mask * (observed - raw)


class FrequencyMixerBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        mixed, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + self.dropout(mixed)
        return tokens + self.dropout(self.ffn(self.norm2(tokens)))


class DualAxisMFGAT(nn.Module):
    """Alternate shared spatial GAT and co-located cross-frequency attention."""

    def __init__(self, width: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.measurement_encoder = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.frequency_embedding = nn.Parameter(torch.empty(1, NF, width))
        nn.init.normal_(self.frequency_embedding, std=0.02)
        self.frequency_mixers = nn.ModuleList(
            [FrequencyMixerBlock(width, heads, dropout) for _ in range(layers + 1)]
        )
        self.spatial_blocks = nn.ModuleList(
            [ResidualGATBlock(width, heads, dropout) for _ in range(layers)]
        )
        self.local_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.graph_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.gate_logits = nn.Parameter(torch.zeros(NF))

    @staticmethod
    def spatial_edges(
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        offsets = (
            torch.arange(NF, device=edge_index.device, dtype=edge_index.dtype)
            * num_nodes
        )
        return (
            edge_index[None, :, :] + offsets[:, None, None]
        ).permute(1, 0, 2).reshape(2, -1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        values = x[:, :NF]
        masks = x[:, NF : 2 * NF]
        tokens = self.measurement_encoder(torch.stack([values, masks], dim=-1))
        tokens = (
            tokens
            + self.geometry_encoder(x[:, 2 * NF : 2 * NF + 5])[:, None, :]
            + self.frequency_embedding
        )
        tokens = self.frequency_mixers[0](tokens)
        local = torch.sigmoid(self.local_head(tokens).squeeze(-1))

        num_nodes = tokens.shape[0]
        replicated_edges = self.spatial_edges(edge_index, num_nodes)
        for spatial, mixer in zip(self.spatial_blocks, self.frequency_mixers[1:]):
            flattened = tokens.permute(1, 0, 2).reshape(NF * num_nodes, -1)
            flattened = spatial(flattened, replicated_edges)
            tokens = flattened.reshape(NF, num_nodes, -1).permute(1, 0, 2)
            tokens = mixer(tokens)
        graph = torch.sigmoid(self.graph_head(tokens).squeeze(-1))
        gate = torch.sigmoid(self.gate_logits)[None, :]
        return local + gate * (graph - local)


class SparseAnchorAttention(nn.Module):
    """Learned attention from every region to nearby observed regions."""

    def __init__(self, width: int, heads: int, neighbors: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = width // heads
        self.neighbors = int(neighbors)
        self.norm1 = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.distance_scale = nn.Parameter(torch.full((heads,), 4.0))
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        anchor_indices = torch.nonzero(anchor_mask, as_tuple=False).flatten()
        if anchor_indices.numel() == 0:
            return tokens

        normalized = self.norm1(tokens)
        anchor_tokens = normalized[anchor_indices]
        anchor_coordinates = coordinates[anchor_indices]
        neighbors = min(self.neighbors, int(anchor_indices.numel()))
        with torch.no_grad():
            distances, nearest = torch.topk(
                torch.cdist(coordinates.float(), anchor_coordinates.float()),
                k=neighbors,
                dim=1,
                largest=False,
                sorted=False,
            )

        selected = anchor_tokens[nearest]
        num_nodes = tokens.shape[0]
        query = self.query(normalized).reshape(
            num_nodes, self.heads, self.head_dim
        )
        keys = self.key(selected).reshape(
            num_nodes, neighbors, self.heads, self.head_dim
        )
        values = self.value(selected).reshape(
            num_nodes, neighbors, self.heads, self.head_dim
        )
        scores = torch.einsum("nhd,nkhd->nkh", query, keys)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores - F.softplus(self.distance_scale)[None, None, :] * distances[:, :, None]
        weights = torch.softmax(scores, dim=1)
        mixed = torch.sum(weights[:, :, :, None] * values, dim=1)
        mixed = mixed.reshape(num_nodes, -1)
        tokens = tokens + self.dropout(self.output(mixed))
        return tokens + self.dropout(self.ffn(self.norm2(tokens)))


class JointSparseAnchorAttention(nn.Module):
    """Attend from each node-frequency token to nearby anchors in every band."""

    def __init__(self, width: int, heads: int, neighbors: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = width // heads
        self.neighbors = int(neighbors)
        self.norm1 = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.distance_scale = nn.Parameter(torch.full((heads,), 4.0))
        self.frequency_bias = nn.Parameter(torch.zeros(NF, NF, heads))
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        anchor_masks: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm1(tokens)
        selected_tokens = []
        selected_distances = []
        selected_frequencies = []
        for source_frequency in range(NF):
            anchor_indices = torch.nonzero(
                anchor_masks[:, source_frequency], as_tuple=False
            ).flatten()
            if anchor_indices.numel() == 0:
                continue
            neighbors = min(self.neighbors, int(anchor_indices.numel()))
            anchor_coordinates = coordinates[anchor_indices]
            with torch.no_grad():
                distances, nearest = torch.topk(
                    torch.cdist(coordinates.float(), anchor_coordinates.float()),
                    k=neighbors,
                    dim=1,
                    largest=False,
                    sorted=False,
                )
            selected_tokens.append(
                normalized[anchor_indices[nearest], source_frequency]
            )
            selected_distances.append(distances)
            selected_frequencies.extend([source_frequency] * neighbors)

        if not selected_tokens:
            return tokens

        anchors = torch.cat(selected_tokens, dim=1)
        distances = torch.cat(selected_distances, dim=1)
        source_frequencies = torch.tensor(
            selected_frequencies, device=tokens.device, dtype=torch.long
        )
        num_nodes, num_frequencies = tokens.shape[:2]
        num_anchors = anchors.shape[1]
        queries = self.query(normalized).reshape(
            num_nodes, num_frequencies, self.heads, self.head_dim
        )
        keys = self.key(anchors).reshape(
            num_nodes, num_anchors, self.heads, self.head_dim
        )
        values = self.value(anchors).reshape(
            num_nodes, num_anchors, self.heads, self.head_dim
        )
        scores = torch.einsum("nfhd,nkhd->nfkh", queries, keys)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores - (
            F.softplus(self.distance_scale)[None, None, None, :]
            * distances[:, None, :, None]
        )
        scores = scores + self.frequency_bias[
            :, source_frequencies, :
        ][None, :, :, :]
        weights = torch.softmax(scores, dim=2)
        mixed = torch.einsum("nfkh,nkhd->nfhd", weights, values)
        mixed = mixed.reshape(num_nodes, num_frequencies, -1)
        tokens = tokens + self.dropout(self.output(mixed))
        return tokens + self.dropout(self.ffn(self.norm2(tokens)))


class AnchorAwareMFGAT(nn.Module):
    """Cross-frequency MFGAT with dynamic observed-region attention edges."""

    def __init__(
        self,
        width: int,
        heads: int,
        layers: int,
        dropout: float,
        anchor_neighbors: int,
    ):
        super().__init__()
        self.measurement_encoder = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.frequency_embedding = nn.Parameter(torch.empty(1, NF, width))
        nn.init.normal_(self.frequency_embedding, std=0.02)
        self.frequency_mixer1 = FrequencyMixerBlock(width, heads, dropout)
        self.anchor_attention = SparseAnchorAttention(
            width, heads, anchor_neighbors, dropout
        )
        self.frequency_mixer2 = FrequencyMixerBlock(width, heads, dropout)
        self.local_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.node_projection = nn.Sequential(
            nn.Linear(NF * width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.blocks = nn.ModuleList(
            [ResidualGATBlock(width, heads, dropout) for _ in range(layers)]
        )
        self.graph_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, NF))
        self.gate_logits = nn.Parameter(torch.zeros(NF))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        values = x[:, :NF]
        masks = x[:, NF : 2 * NF]
        geometry = x[:, 2 * NF : 2 * NF + 5]
        tokens = self.measurement_encoder(torch.stack([values, masks], dim=-1))
        tokens = (
            tokens
            + self.geometry_encoder(geometry)[:, None, :]
            + self.frequency_embedding
        )
        tokens = self.frequency_mixer1(tokens)
        coordinates = geometry[:, :2]
        tokens = torch.stack(
            [
                self.anchor_attention(
                    tokens[:, frequency],
                    coordinates,
                    masks[:, frequency] > 0.5,
                )
                for frequency in range(NF)
            ],
            dim=1,
        )
        tokens = self.frequency_mixer2(tokens)
        local = torch.sigmoid(self.local_head(tokens).squeeze(-1))
        hidden = self.node_projection(tokens.reshape(tokens.shape[0], -1))
        for block in self.blocks:
            hidden = block(hidden, edge_index)
        graph = torch.sigmoid(self.graph_head(hidden))
        gate = torch.sigmoid(self.gate_logits)[None, :]
        return local + gate * (graph - local)


class JointAnchorAwareMFGAT(nn.Module):
    """MFGAT with direct spatial attention to observations from every band."""

    def __init__(
        self,
        width: int,
        heads: int,
        layers: int,
        dropout: float,
        anchor_neighbors: int,
    ):
        super().__init__()
        self.measurement_encoder = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.frequency_embedding = nn.Parameter(torch.empty(1, NF, width))
        nn.init.normal_(self.frequency_embedding, std=0.02)
        self.frequency_mixer1 = FrequencyMixerBlock(width, heads, dropout)
        self.anchor_attention = JointSparseAnchorAttention(
            width, heads, anchor_neighbors, dropout
        )
        self.frequency_mixer2 = FrequencyMixerBlock(width, heads, dropout)
        self.local_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.node_projection = nn.Sequential(
            nn.Linear(NF * width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.blocks = nn.ModuleList(
            [ResidualGATBlock(width, heads, dropout) for _ in range(layers)]
        )
        self.graph_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, NF))
        self.gate_logits = nn.Parameter(torch.zeros(NF))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        values = x[:, :NF]
        masks = x[:, NF : 2 * NF]
        geometry = x[:, 2 * NF : 2 * NF + 5]
        tokens = self.measurement_encoder(torch.stack([values, masks], dim=-1))
        tokens = (
            tokens
            + self.geometry_encoder(geometry)[:, None, :]
            + self.frequency_embedding
        )
        tokens = self.frequency_mixer1(tokens)
        tokens = self.anchor_attention(
            tokens,
            geometry[:, :2],
            masks > 0.5,
        )
        tokens = self.frequency_mixer2(tokens)
        local = torch.sigmoid(self.local_head(tokens).squeeze(-1))
        hidden = self.node_projection(tokens.reshape(tokens.shape[0], -1))
        for block in self.blocks:
            hidden = block(hidden, edge_index)
        graph = torch.sigmoid(self.graph_head(hidden))
        gate = torch.sigmoid(self.gate_logits)[None, :]
        return local + gate * (graph - local)


class HybridAnchorAwareMFGAT(nn.Module):
    """Fuse same-band and cross-frequency sparse-anchor attention."""

    def __init__(
        self,
        width: int,
        heads: int,
        layers: int,
        dropout: float,
        anchor_neighbors: int,
        joint_anchor_neighbors: int,
    ):
        super().__init__()
        self.measurement_encoder = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.frequency_embedding = nn.Parameter(torch.empty(1, NF, width))
        nn.init.normal_(self.frequency_embedding, std=0.02)
        self.frequency_mixer1 = FrequencyMixerBlock(width, heads, dropout)
        self.same_band_attention = SparseAnchorAttention(
            width, heads, anchor_neighbors, dropout
        )
        self.joint_attention = JointSparseAnchorAttention(
            width, heads, joint_anchor_neighbors, dropout
        )
        self.anchor_gate_logits = nn.Parameter(torch.zeros(NF))
        self.frequency_mixer2 = FrequencyMixerBlock(width, heads, dropout)
        self.local_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.node_projection = nn.Sequential(
            nn.Linear(NF * width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.blocks = nn.ModuleList(
            [ResidualGATBlock(width, heads, dropout) for _ in range(layers)]
        )
        self.graph_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, NF))
        self.output_gate_logits = nn.Parameter(torch.zeros(NF))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        values = x[:, :NF]
        masks = x[:, NF : 2 * NF]
        geometry = x[:, 2 * NF : 2 * NF + 5]
        tokens = self.measurement_encoder(torch.stack([values, masks], dim=-1))
        tokens = (
            tokens
            + self.geometry_encoder(geometry)[:, None, :]
            + self.frequency_embedding
        )
        tokens = self.frequency_mixer1(tokens)
        coordinates = geometry[:, :2]
        same_band = torch.stack(
            [
                self.same_band_attention(
                    tokens[:, frequency],
                    coordinates,
                    masks[:, frequency] > 0.5,
                )
                for frequency in range(NF)
            ],
            dim=1,
        )
        joint = self.joint_attention(tokens, coordinates, masks > 0.5)
        anchor_gate = torch.sigmoid(self.anchor_gate_logits)[None, :, None]
        tokens = same_band + anchor_gate * (joint - tokens)
        tokens = self.frequency_mixer2(tokens)
        local = torch.sigmoid(self.local_head(tokens).squeeze(-1))
        hidden = self.node_projection(tokens.reshape(tokens.shape[0], -1))
        for block in self.blocks:
            hidden = block(hidden, edge_index)
        graph = torch.sigmoid(self.graph_head(hidden))
        output_gate = torch.sigmoid(self.output_gate_logits)[None, :]
        return local + output_gate * (graph - local)


class GroupedJointAnchorMFGAT(nn.Module):
    """Joint anchors followed by separate sub-6 and mmWave spatial GATs."""

    def __init__(
        self,
        width: int,
        heads: int,
        layers: int,
        dropout: float,
        anchor_neighbors: int,
    ):
        super().__init__()
        self.measurement_encoder = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.frequency_embedding = nn.Parameter(torch.empty(1, NF, width))
        nn.init.normal_(self.frequency_embedding, std=0.02)
        self.frequency_mixer1 = FrequencyMixerBlock(width, heads, dropout)
        self.anchor_attention = JointSparseAnchorAttention(
            width, heads, anchor_neighbors, dropout
        )
        self.frequency_mixer2 = FrequencyMixerBlock(width, heads, dropout)
        self.sub6_local_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.mmwave_local_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 1)
        )
        self.sub6_projection = nn.Sequential(
            nn.Linear(len(SUB6) * width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.mmwave_projection = nn.Sequential(
            nn.Linear(len(MMWAVE) * width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.sub6_blocks = nn.ModuleList(
            [ResidualGATBlock(width, heads, dropout) for _ in range(layers)]
        )
        self.mmwave_blocks = nn.ModuleList(
            [ResidualGATBlock(width, heads, dropout) for _ in range(layers)]
        )
        self.sub6_graph_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, len(SUB6))
        )
        self.mmwave_graph_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, len(MMWAVE))
        )
        self.gate_logits = nn.Parameter(torch.zeros(NF))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        values = x[:, :NF]
        masks = x[:, NF : 2 * NF]
        geometry = x[:, 2 * NF : 2 * NF + 5]
        tokens = self.measurement_encoder(torch.stack([values, masks], dim=-1))
        tokens = (
            tokens
            + self.geometry_encoder(geometry)[:, None, :]
            + self.frequency_embedding
        )
        tokens = self.frequency_mixer1(tokens)
        tokens = self.anchor_attention(tokens, geometry[:, :2], masks > 0.5)
        tokens = self.frequency_mixer2(tokens)

        sub6_tokens = tokens[:, SUB6]
        mmwave_tokens = tokens[:, MMWAVE]
        local = torch.cat(
            [
                torch.sigmoid(self.sub6_local_head(sub6_tokens).squeeze(-1)),
                torch.sigmoid(self.mmwave_local_head(mmwave_tokens).squeeze(-1)),
            ],
            dim=1,
        )
        sub6_hidden = self.sub6_projection(
            sub6_tokens.reshape(sub6_tokens.shape[0], -1)
        )
        mmwave_hidden = self.mmwave_projection(
            mmwave_tokens.reshape(mmwave_tokens.shape[0], -1)
        )
        for block in self.sub6_blocks:
            sub6_hidden = block(sub6_hidden, edge_index)
        for block in self.mmwave_blocks:
            mmwave_hidden = block(mmwave_hidden, edge_index)
        graph = torch.cat(
            [
                torch.sigmoid(self.sub6_graph_head(sub6_hidden)),
                torch.sigmoid(self.mmwave_graph_head(mmwave_hidden)),
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.gate_logits)[None, :]
        return local + gate * (graph - local)


class LDPLFrequencyAwareMFGAT(nn.Module):
    """Frequency-token MFGAT with an anchor-fitted LDPL input feature."""

    def __init__(self, width: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.measurement_encoder = nn.Sequential(
            nn.Linear(3, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.frequency_embedding = nn.Parameter(torch.empty(1, NF, width))
        nn.init.normal_(self.frequency_embedding, std=0.02)
        self.frequency_mixer = FrequencyMixerBlock(width, heads, dropout)
        self.local_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.node_projection = nn.Sequential(
            nn.Linear(NF * width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.blocks = nn.ModuleList(
            [ResidualGATBlock(width, heads, dropout) for _ in range(layers)]
        )
        self.graph_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, NF))
        self.gate_logits = nn.Parameter(torch.zeros(NF))

    @staticmethod
    def ldpl_from_visible_anchors(x: torch.Tensor) -> torch.Tensor:
        distance = x[:, 2 * NF + 2].clamp_min(1e-3)
        design = torch.stack([torch.ones_like(distance), torch.log10(distance)], dim=1)
        values = x[:, :NF]
        masks = x[:, NF : 2 * NF] > 0.5
        priors = []
        regularizer = x.new_tensor([[1e-6, 0.0], [0.0, 1e-3]])
        for frequency in range(NF):
            active = masks[:, frequency]
            if int(active.sum()) >= 2:
                active_design = design[active]
                active_values = values[active, frequency]
                coefficients = torch.linalg.solve(
                    active_design.T @ active_design + regularizer,
                    active_design.T @ active_values,
                )
                prior = design @ coefficients
            elif active.any():
                prior = values[active, frequency].mean().expand_as(distance)
            else:
                prior = x.new_full(distance.shape, 0.5)
            priors.append(prior.clamp(0.0, 1.0))
        return torch.stack(priors, dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        values = x[:, :NF]
        masks = x[:, NF : 2 * NF]
        ldpl = self.ldpl_from_visible_anchors(x)
        measurement = torch.stack([values, masks, ldpl], dim=-1)
        tokens = self.measurement_encoder(measurement)
        tokens = (
            tokens
            + self.geometry_encoder(x[:, 2 * NF : 2 * NF + 5])[:, None, :]
            + self.frequency_embedding
        )
        tokens = self.frequency_mixer(tokens)
        local = torch.sigmoid(self.local_head(tokens).squeeze(-1))
        hidden = self.node_projection(tokens.reshape(tokens.shape[0], -1))
        for block in self.blocks:
            hidden = block(hidden, edge_index)
        graph = torch.sigmoid(self.graph_head(hidden))
        gate = torch.sigmoid(self.gate_logits)[None, :]
        return local + gate * (graph - local)


def mask_training_anchors(
    x: torch.Tensor,
    train_mask: torch.Tensor,
    drop_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hide sparse labels from the input and return the reconstruction mask."""

    hide = train_mask & (torch.rand_like(train_mask, dtype=torch.float32) < drop_probability)
    for frequency in range(NF):
        active = torch.nonzero(train_mask[:, frequency], as_tuple=False).flatten()
        if active.numel() and not hide[:, frequency].any():
            selected = active[torch.randint(active.numel(), (1,), device=active.device)]
            hide[selected, frequency] = True
        if active.numel() > 1 and torch.all(hide[active, frequency]):
            selected = active[torch.randint(active.numel(), (1,), device=active.device)]
            hide[selected, frequency] = False

    masked_x = x.clone()
    sparse = masked_x[:, :NF]
    masks = masked_x[:, NF : 2 * NF]
    sparse[hide] = 0.0
    masks[hide] = 0.0
    return masked_x, hide


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "huber":
        return frequency_balanced_loss(pred, target, mask)
    losses = []
    for frequency in range(NF):
        active = mask[:, frequency]
        if active.any():
            losses.append(
                F.mse_loss(pred[active, frequency], target[active, frequency])
            )
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def build_model(args: argparse.Namespace) -> tuple[nn.Module, str]:
    if args.architecture == "frequency_token":
        core = FrequencyAwareMFGAT(
            args.hidden_dim,
            args.num_heads,
            args.num_layers,
            args.dropout,
        )
        architecture = "frequency_token_attention_plus_residual_spatial_gat"
    elif args.architecture == "frequency_spatial":
        core = DualAxisMFGAT(
            args.hidden_dim,
            args.num_heads,
            args.num_layers,
            args.dropout,
        )
        architecture = "alternating_cross_frequency_attention_and_spatial_gat"
    elif args.architecture == "frequency_anchor":
        core = AnchorAwareMFGAT(
            args.hidden_dim,
            args.num_heads,
            args.num_layers,
            args.dropout,
            args.anchor_neighbors,
        )
        architecture = "cross_frequency_attention_plus_dynamic_sparse_anchor_gat"
    elif args.architecture == "frequency_joint_anchor":
        core = JointAnchorAwareMFGAT(
            args.hidden_dim,
            args.num_heads,
            args.num_layers,
            args.dropout,
            args.anchor_neighbors,
        )
        architecture = "joint_cross_frequency_sparse_anchor_attention_plus_spatial_gat"
    elif args.architecture == "frequency_hybrid_anchor":
        core = HybridAnchorAwareMFGAT(
            args.hidden_dim,
            args.num_heads,
            args.num_layers,
            args.dropout,
            args.anchor_neighbors,
            args.joint_anchor_neighbors,
        )
        architecture = (
            "hybrid_same_band_and_cross_frequency_sparse_anchor_attention_"
            "plus_spatial_gat"
        )
    elif args.architecture == "frequency_grouped_joint_anchor":
        core = GroupedJointAnchorMFGAT(
            args.hidden_dim,
            args.num_heads,
            args.num_layers,
            args.dropout,
            args.anchor_neighbors,
        )
        architecture = (
            "joint_cross_frequency_sparse_anchor_attention_plus_separate_"
            "sub6_and_mmwave_spatial_gats"
        )
    elif args.architecture == "frequency_ldpl":
        core = LDPLFrequencyAwareMFGAT(
            args.hidden_dim,
            args.num_heads,
            args.num_layers,
            args.dropout,
        )
        architecture = "anchor_fitted_ldpl_plus_frequency_attention_and_spatial_gat"
    else:
        core = PureGroupedMFGAT(
            args.hidden_dim,
            args.num_heads,
            args.num_layers,
            args.dropout,
        )
        architecture = "sub6_mmwave_grouped_residual_spatial_gat"
    return AnchorPreservingMFGAT(core, args.consistency_alpha), architecture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=[
            "frequency_token",
            "frequency_spatial",
            "frequency_anchor",
            "frequency_joint_anchor",
            "frequency_hybrid_anchor",
            "frequency_grouped_joint_anchor",
            "frequency_ldpl",
            "frequency_grouped",
        ],
        default="frequency_token",
    )
    parser.add_argument("--cities", default="Davis,Woodland,Vacaville")
    parser.add_argument("--max-graphs", type=int, default=1160)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--anchor-dropout", type=float, default=0.50)
    parser.add_argument("--anchor-neighbors", type=int, default=32)
    parser.add_argument("--joint-anchor-neighbors", type=int, default=4)
    parser.add_argument("--consistency-alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-type", choices=["huber", "mse"], default="huber")
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--target-val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Fixed validation/test split seed; defaults to --seed.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--figure-freq-index", type=int, default=1)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.anchor_dropout < 1.0:
        raise ValueError("anchor_dropout must be in (0, 1)")
    if args.hidden_dim % args.num_heads:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if args.anchor_neighbors < 1:
        raise ValueError("anchor_neighbors must be positive")
    if args.joint_anchor_neighbors < 1:
        raise ValueError("joint_anchor_neighbors must be positive")
    if not 0.0 <= args.consistency_alpha <= 1.0:
        raise ValueError("consistency_alpha must be in [0, 1]")
    if args.architecture in {
        "frequency_ldpl",
        "frequency_anchor",
        "frequency_joint_anchor",
        "frequency_hybrid_anchor",
        "frequency_grouped_joint_anchor",
    } and args.batch_size != 1:
        raise ValueError(f"{args.architecture} requires batch_size=1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cities = [city.strip() for city in args.cities.split(",") if city.strip()]
    split_seed = args.seed if args.split_seed is None else args.split_seed
    examples = load_examples(
        args.graph_root,
        cities,
        args.max_graphs,
        split_seed,
        "no_idw",
        args.target_cache_root,
        args.target_val_fraction,
        False,
    )
    if not examples:
        raise RuntimeError(f"No graphs under {args.graph_root}")
    if examples[0].x.shape[1] != 2 * NF + 5:
        raise RuntimeError(
            f"Expected no-IDW input width {2 * NF + 5}, got {examples[0].x.shape[1]}"
        )

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else "cpu"
    )
    model, architecture = build_model(args)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.05,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history: list[dict[str, object]] = []
    print(
        f"[v7 masked-anchor] architecture={args.architecture} graphs={len(examples)} "
        f"features={examples[0].x.shape[1]} anchor_dropout={args.anchor_dropout} "
        f"device={device}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        model.train()
        losses: list[float] = []
        data_losses: list[float] = []
        smooth_losses: list[float] = []
        hidden_fractions: list[float] = []
        started = time.time()

        for x, target, train_mask, edge in graph_batches(examples, args.batch_size):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            train_mask = train_mask.to(device, non_blocking=True)
            edge = edge.to(device, non_blocking=True)
            masked_x, reconstruction_mask = mask_training_anchors(
                x, train_mask, args.anchor_dropout
            )
            pred = model(masked_x, edge)
            data_loss = reconstruction_loss(
                pred, target, reconstruction_mask, args.loss_type
            )
            smooth_loss = (
                graph_smoothness(pred, edge)
                if args.smoothness_weight > 0.0
                else pred.sum() * 0.0
            )
            loss = data_loss + args.smoothness_weight * smooth_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            data_losses.append(float(data_loss.detach().cpu()))
            smooth_losses.append(float(smooth_loss.detach().cpu()))
            hidden_fractions.append(
                float(reconstruction_mask.sum().cpu())
                / max(float(train_mask.sum().cpu()), 1.0)
            )

        scheduler.step()
        record: dict[str, object] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "data_loss": float(np.mean(data_losses)),
            "smoothness_loss": float(np.mean(smooth_losses)),
            "hidden_anchor_fraction": float(np.mean(hidden_fractions)),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
        }
        should_eval = (
            epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        )
        if should_eval:
            val = evaluate(model, examples, device, "val")
            record["val"] = val
            if val["rmse_db"] < best_val:
                best_val = float(val["rmse_db"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, args.out_dir / "best_model.pt")
                stale = 0
            else:
                stale += 1
        history.append(record)
        (args.out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        if should_eval and (
            epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs
        ):
            print(
                f"[epoch {epoch:04d}] loss={record['loss']:.6f} "
                f"hidden={record['hidden_anchor_fraction']:.3f} "
                f"val={val['rmse_db']:.3f}dB sec={record['seconds']:.1f}",
                flush=True,
            )
        if should_eval and args.patience > 0 and stale >= args.patience:
            print(f"[early stop] epoch={epoch} stale_evaluations={stale}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was produced")
    model.load_state_dict(best_state)
    torch.save(best_state, args.out_dir / "best_model.pt")
    result: dict[str, object] = {
        "method": "MFGAT",
        "protocol": "stage1_target_environment_masked_anchor_v7",
        "architecture": architecture,
        "input": "sparse_node_values_masks_geometry_no_idw",
        "training_supervision": "randomly_hidden_sparse_node_frequency_anchors_only",
        "validation_supervision": "disjoint_sparse_validation_pixels_only",
        "dense_target_usage": "final_evaluation_only",
        "inference_consistency": (
            "region_anchor_blend_alpha=" f"{args.consistency_alpha:g}"
        ),
        "target_schema": examples[0].target_schema,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "num_examples": len(examples),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_rmse_db": best_val,
        "val": evaluate(model, examples, device, "val"),
        "test_evaluated": not args.skip_test_evaluation,
    }
    if not args.skip_test_evaluation:
        result["test"] = evaluate(model, examples, device, "test")
    (args.out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if not args.skip_test_evaluation:
        make_figure(
            model,
            examples[0],
            device,
            args.out_dir / "prediction_example.png",
            "MFGAT (masked-anchor)",
            args.figure_freq_index,
        )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
