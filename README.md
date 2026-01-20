# EstateX: AI-Powered Real Estate Discovery Engine

This repository contains the core recommendation system powering the EstateX discovery feed. It utilizes a two-stage ranking architecture—incorporating Computer Vision (CLIP) for multi-modal latent space retrieval and Gradient Boosted Decision Trees (GBDT) for precision ranking—to surface property listings based on gesture-based (swipe) interaction patterns.

## Table of Contents

- [EstateX: AI-Powered Real Estate Discovery Engine](#estatex-ai-powered-real-estate-discovery-engine)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [System Architecture](#system-architecture)
  - [Components](#components)
    - [EstateX API (Home Mixer Equivalent)](#estatex-api-home-mixer-equivalent)
    - [Ingestion Engine](#ingestion-engine)
      - [Image Pipeline](#image-pipeline)
    - [Light-Ranker (Candidate Generation)](#light-ranker-candidate-generation)
      - [Retrieval Architecture](#retrieval-architecture)
      - [Functionality](#functionality)
    - [Heavy-Ranker (Precision Scoring)](#heavy-ranker-precision-scoring)
      - [Model Details](#model-details)
      - [Predicted Actions](#predicted-actions)
      - [Features](#features)
    - [Diversity \& Anti-Stagnation Scorer](#diversity--anti-stagnation-scorer)
  - [How It Works](#how-it-works)
    - [Pipeline Stages](#pipeline-stages)
    - [Engagement-Weighted Ranking](#engagement-weighted-ranking)
    - [Filtering](#filtering)
      - [Pre-Scoring Filters](#pre-scoring-filters)
      - [Post-Selection Filters](#post-selection-filters)
  - [Key Design Decisions](#key-design-decisions)
    - [1. Unified Visual Embedding](#1-unified-visual-embedding)
    - [2. Two-Stage Ranking Architecture](#2-two-stage-ranking-architecture)
    - [3. Multi-Action Prediction](#3-multi-action-prediction)
    - [4. Real-Time Preference Adaptation](#4-real-time-preference-adaptation)
    - [5. Composable Pipeline Design](#5-composable-pipeline-design)

---

## Overview

The EstateX algorithm retrieves, ranks, and filters property listings from a global corpus using a neural discovery pipeline. The system replaces traditional parametric search with a reinforcement learning loop driven by user interaction (swiping).

1. **Candidate Generation**: Retrieval of ~500 listings from a 512-dimensional latent space.
2. **Precision Ranking**: Heavy-duty scoring of candidates using a GBDT model that predicts engagement probabilities based on historical interaction sequences.

The engine uses **CLIP (Contrastive Language-Image Pre-training)** to process imagery, ensuring that architectural styles and visual features are mathematically represented without manual tagging.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PROPERTY DISCOVERY REQUEST                                │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         ESTATEX API                                         │
│                                    (Orchestration Layer)                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                   QUERY HYDRATION                                   │   │
│   │  ┌──────────────────────────┐    ┌──────────────────────────────────────────────┐   │   │
│   │  │ User Preference Vector   │    │ Geospatial Constraints                       │   │   │
│   │  │ (Interaction History)    │    │ (Polygon/Radius Filters)                     │   │   │
│   │  └──────────────────────────┘    └──────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                  CANDIDATE SOURCES                                  │   │
│   │         ┌─────────────────────────────┐    ┌────────────────────────────────┐       │   │
│   │         │       LIGHT-RANKER          │    │      DISCOVERY RETRIEVAL       │       │   │
│   │         │      (Vector Search)        │    │    (Collaborative Filtering)   │       │   │
│   │         │                             │    │                                │       │   │
│   │         │  HNSW Similarity Search     │    │  In-network/Out-of-network     │       │   │
│   │         │  in 512-dim latent space    │    │  trending properties           │       │   │
│   │         └─────────────────────────────┘    └────────────────────────────────┘       │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                      HYDRATION                                      │   │
│   │  Retrieval of listing metadata and feature augmentation (Price/SQFT/Neighborhood)   │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                       SCORING                                       │   │
│   │  ┌──────────────────────────┐                                                       │   │
│   │  │  Heavy-Ranker (GBDT)     │    Predicts probability of interaction:               │   │
│   │  │  (Precision Scorer)      │    P(right_swipe), P(save), P(share)                  │   │
│   │  └──────────────────────────┘                                                       │   │
│   │               │                                                                     │   │
│   │               ▼                                                                     │   │
│   │  ┌──────────────────────────┐                                                       │   │
│   │  │  MMR Diversity Scorer    │    Applies visual/geospatial diversity                │   │
│   │  │  (Anti-Filter Bubble)    │    to prevent feed stagnation                         │   │
│   │  └──────────────────────────┘                                                       │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                      SELECTION                                      │   │
│   │               Sorting by composite score for real-time swipe delivery               │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     RANKED PROPERTY FEED                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### EstateX API (Home Mixer Equivalent)

**Location:** `backend/api/`

The orchestration layer that assembles the EstateX discovery feed. It coordinates retrieval, ranking, filtering, and session updates using a composable candidate pipeline.

The API exposes a real-time discovery endpoint that returns ranked property listings for a given user session.

| Stage | Description |
|------|-------------|
| Query Hydrators | Fetch user preference vectors and session interaction history |
| Sources | Retrieve candidates from Light-Ranker and Discovery Retrieval |
| Hydrators | Enrich listings with metadata and derived features |
| Filters | Remove ineligible or redundant listings |
| Scorers | Predict engagement and compute final ranking scores |
| Selector | Sort by score and select top K listings |
| Side Effects | Persist session state and update preference vectors |

User interactions (swipes, saves, shares) are streamed back into Redis to continuously refine the ranking context mid-session.

---

### Ingestion Engine

**Location:** `backend/data/`

A real-time data acquisition and normalization system responsible for maintaining the global property corpus.

It performs the following functions:

- Scrapes listings via Playwright-driven headless browser instances
- Normalizes raw property metadata into a structured relational schema
- Processes and stores price, location, and neighborhood attributes
- Handles image ingestion for downstream embedding generation

#### Image Pipeline
- Encodes property images using **CLIP ViT-B/32**
- Produces **512-dimensional visual embeddings**
- Ensures architectural style and visual features are learned implicitly, without manual tagging

---

### Light-Ranker (Candidate Generation)

**Location:** `backend/database/`

A high-throughput retrieval layer responsible for narrowing the global corpus to a manageable candidate set.

#### Retrieval Architecture
- **Vector Database:** Qdrant
- **ANN Algorithm:** Hierarchical Navigable Small World (HNSW)
- **Similarity Metric:** Cosine similarity
- **Embedding Space:** 512-dimensional CLIP latent space

#### Functionality
- Computes a weighted centroid of the user’s preference vector
- Retrieves approximately **500 visually relevant properties**
- Optimized for high recall and low latency (<100ms)

This stage prioritizes discovery breadth over precision.

---

### Heavy-Ranker (Precision Scoring)

**Location:** `backend/core/`

The precision ranking model responsible for ordering candidates surfaced by the Light-Ranker.

#### Model Details
- **Algorithm:** Gradient Boosted Decision Trees (GBDT)
- **Objective:** Predict probabilities of multiple user interactions

#### Predicted Actions

```
Predictions:
├── P(right_swipe)
├── P(save)
├── P(share)
├── P(dwell_time > 10s)
└── P(skip)
```


#### Features
- Visual latent embeddings
- Price deviation from user preference
- Geospatial proximity
- Session-level interaction history
- Temporal listing freshness

---

### Diversity & Anti-Stagnation Scorer

**Location:** `backend/core/mmr/`

A post-scoring diversification layer applied after precision ranking.

- Uses **Maximal Marginal Relevance (MMR)**
- Penalizes visually redundant listings
- Enforces geographic and architectural diversity
- Prevents filter bubbles within swipe sessions

---

## How It Works

### Pipeline Stages

1. **Query Hydration**  
   Load the user’s current preference vector, swipe history, and geospatial constraints.

2. **Candidate Sourcing**  
   Retrieve candidates from:
   - **Light-Ranker:** CLIP-based vector similarity search
   - **Discovery Retrieval:** Trending and collaborative-filtered properties

3. **Candidate Hydration**  
   Enrich listings with:
   - Price, square footage, and neighborhood data
   - Distance calculations
   - Image embeddings
   - Listing freshness indicators

4. **Pre-Scoring Filters**  
   Remove listings that are:
   - Outside geospatial constraints
   - Previously shown in-session
   - Duplicates or visually near-identical
   - Missing required metadata
   - Price-incompatible with user constraints

5. **Scoring**  
   Apply scorers sequentially:
   - **Heavy-Ranker (GBDT):** Predict engagement probabilities
   - **Engagement Weighted Scorer:** Combine predictions into a composite score
   - **MMR Diversity Scorer:** Enforce session diversity

6. **Selection**  
   Sort by final score and select top K properties for swipe delivery.

7. **Side Effects**  
   Persist interactions and update the user preference vector in Redis for immediate feedback.

---

### Engagement-Weighted Ranking

The final score for each property is computed as:

```
Final Score =
  1.0 × P(right_swipe)
+ 5.0 × P(save)
+ 3.0 × P(share)
+ 2.0 × P(dwell_time > 10s)
```

Higher-intent actions (saving and sharing) are weighted more heavily to optimize long-term user value rather than short-term novelty.

---

### Filtering

Filters are applied at two stages:

#### Pre-Scoring Filters

| Filter | Purpose |
|------|---------|
| `GeoConstraintFilter` | Enforce polygon/radius boundaries |
| `DuplicateListingFilter` | Remove duplicate property IDs |
| `SessionSeenFilter` | Remove listings already shown |
| `MetadataCompletenessFilter` | Drop incomplete listings |
| `PriceRangeFilter` | Enforce user price tolerance |
| `VisualNearDuplicateFilter` | Remove visually redundant properties |

#### Post-Selection Filters

| Filter | Purpose |
|------|---------|
| `MMRRedundancyFilter` | Final diversity enforcement |
| `AvailabilityFilter` | Remove delisted or inactive properties |

---

## Key Design Decisions

### 1. Unified Visual Embedding
CLIP enables architectural understanding directly from imagery, eliminating manual labeling and brittle heuristics.

### 2. Two-Stage Ranking Architecture
Separating retrieval and ranking allows the system to scale globally while maintaining sub-200ms latency.

### 3. Multi-Action Prediction
The Heavy-Ranker predicts multiple interaction types instead of a single relevance score, balancing discovery with intent.

### 4. Real-Time Preference Adaptation
Session interactions are immediately reflected in the preference vector without retraining or batch updates.

### 5. Composable Pipeline Design
Each stage is modular, allowing rapid experimentation with new scorers, filters, and retrieval strategies.
