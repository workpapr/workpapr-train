import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import type { TraceStep, TraceAction, InvestigationTrace } from "@workpapr/core/training/types";
import type { FindingType, Severity } from "@workpapr/core/scanner/types";
import type { TraceHook } from "@workpapr/core/analyzer/trace-hook";

export class TraceRecorder implements TraceHook {
  private traces: Map<string, InvestigationTrace> = new Map();
  private outputPath: string;
  private enabled: boolean;

  constructor(rootDir: string, enabled = true) {
    this.enabled = enabled;
    const dir = path.join(rootDir, ".workpapr", "training-data");
    if (enabled) {
      fs.mkdirSync(dir, { recursive: true });
    }
    this.outputPath = path.join(dir, "traces.jsonl");
  }

  startTrace(
    findingId: string,
    findingType: FindingType,
    findingSeverity: Severity,
    file: string
  ): string {
    if (!this.enabled) return "";

    const traceId = `tr-${createHash("sha256").update(`${findingId}-${Date.now()}`).digest("hex").slice(0, 12)}`;

    this.traces.set(traceId, {
      id: traceId,
      findingId,
      findingType,
      findingSeverity,
      file,
      steps: [],
      startedAt: new Date().toISOString(),
    });

    return traceId;
  }

  recordStep(
    traceId: string,
    action: TraceAction,
    target: string,
    resultSummary: string,
    decision: string,
    durationMs?: number
  ): void {
    if (!this.enabled) return;

    const trace = this.traces.get(traceId);
    if (!trace) return;

    trace.steps.push({
      action,
      target,
      resultSummary,
      decision,
      timestamp: new Date().toISOString(),
      durationMs,
    });
  }

  /**
   * Record a pipeline-level step that applies to all active traces.
   * Used when the pipeline does a batch operation (e.g., FP filtering all findings at once).
   */
  recordPipelineStep(
    action: TraceAction,
    target: string,
    resultSummary: string,
    decision: string,
    durationMs?: number
  ): void {
    if (!this.enabled) return;

    for (const trace of this.traces.values()) {
      if (!trace.completedAt) {
        this.recordStep(trace.id, action, target, resultSummary, decision, durationMs);
      }
    }
  }

  completeTrace(traceId: string, finalDisposition?: string): void {
    if (!this.enabled) return;

    const trace = this.traces.get(traceId);
    if (!trace) return;

    trace.completedAt = new Date().toISOString();
    trace.finalDisposition = finalDisposition;

    // Append to JSONL
    const line = JSON.stringify(trace) + "\n";
    fs.appendFileSync(this.outputPath, line, "utf-8");
  }

  completeAllTraces(): void {
    if (!this.enabled) return;

    for (const [traceId, trace] of this.traces) {
      if (!trace.completedAt) {
        this.completeTrace(traceId);
      }
    }
  }

  getTrace(traceId: string): InvestigationTrace | undefined {
    return this.traces.get(traceId);
  }

  get activeTraceCount(): number {
    let count = 0;
    for (const trace of this.traces.values()) {
      if (!trace.completedAt) count++;
    }
    return count;
  }
}

/**
 * Load traces from disk for export/analysis.
 */
export function loadTraces(rootDir: string): InvestigationTrace[] {
  const tracePath = path.join(rootDir, ".workpapr", "training-data", "traces.jsonl");
  if (!fs.existsSync(tracePath)) return [];

  try {
    const raw = fs.readFileSync(tracePath, "utf-8");
    return raw
      .split("\n")
      .filter((line) => line.trim())
      .map((line) => JSON.parse(line) as InvestigationTrace);
  } catch {
    return [];
  }
}
