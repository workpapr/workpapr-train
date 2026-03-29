import type { Severity, ScanResult } from "@workpapr/core/scanner/types";

// --- MLX Training configuration ---

export interface SFTConfig {
  baseModel: string;
  dataDir: string;
  outputDir: string;
  iterations: number;
  learningRate: number;
  loraRank: number;
  loraLayers: number;
}

export interface DPOConfig {
  baseModel: string;
  sftAdapterPath: string;
  dataPath: string;
  outputDir: string;
  iterations: number;
  learningRate: number;
  beta: number;
  loraLayers: number;
}

export interface TrainingProgress {
  phase: "sft" | "dpo";
  iteration: number;
  totalIterations: number;
  loss: number;
  learningRate: number;
  iterPerSec: number;
  peakMemGB: number;
}

export interface MLXPreflightResult {
  pythonPath: string;
  pythonVersion: string;
  mlxVersion: string;
  ok: boolean;
  error?: string;
}

// --- Shared data types (used by both train and demo for data conversion) ---

export interface GroundTruth {
  severity: Severity;
  is_false_positive: boolean;
  reasoning: string;
}

export interface AIPrediction {
  severity: Severity;
  is_false_positive: boolean;
  reasoning: string;
  model: string;
}

export interface DemoFinding {
  scan: ScanResult;
  ground_truth: GroundTruth;
  ai_prediction?: AIPrediction;
  trained_prediction?: AIPrediction;
}

export interface Correction {
  finding_id: string;
  field: "severity" | "fp_flag";
  ai_value: string;
  correct_value: string;
  context: string;
  reasoning: string;
}

export interface StyleEdit {
  sectionId: string;
  sectionTitle: string;
  original: string;
  edited: string;
  diffSummary: string;
}

export interface PrescribedWorkflow {
  findingId: string;
  file: string;
  findingType: string;
  steps: Array<{
    action: string;
    target: string;
    expectedResult: string;
    decision: string;
  }>;
  disposition: string;
}
