import { spawn, exec, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import type {
  DemoFinding,
  Correction,
  SFTConfig,
  DPOConfig,
  TrainingProgress,
  MLXPreflightResult,
  StyleEdit,
  PrescribedWorkflow,
} from "../types.js";

const ANALYSIS_SYSTEM_PROMPT = `You are an AI code auditor analyzing security and compliance findings in a codebase.

For each finding, assess:
1. What severity should it be? (critical, high, medium, low)
2. Is this a false positive? (true/false)
3. Explain your reasoning in 1-2 sentences.

Respond ONLY in JSON: {"severity": "critical|high|medium|low", "is_false_positive": true|false, "reasoning": "..."}`;

const WORKPAPER_SYSTEM_PROMPT = `You are a senior financial services auditor writing formal audit workpaper sections. Use proper audit language: "observation" not "finding", "exposure" not "risk", "remediation action" not "fix". Cite specific evidence with line numbers and code snippets. Use passive voice for observations.`;

const INVESTIGATION_SYSTEM_PROMPT = `You are a senior auditor. When given an audit finding, prescribe the exact investigation steps. Respond in JSON: {"steps": [{"action": "...", "target": "...", "expectedResult": "...", "decision": "..."}], "disposition": "confirmed|false-positive|accepted-risk|needs-remediation"}`;

// --- Preflight ---

export async function checkMLXPrereqs(): Promise<MLXPreflightResult> {
  const result: MLXPreflightResult = {
    pythonPath: "",
    pythonVersion: "",
    mlxVersion: "",
    ok: false,
  };

  // Find python
  const pythonCmd = await findPython();
  if (!pythonCmd) {
    result.error = "Python 3.10+ not found. Install from python.org or via brew install python";
    return result;
  }
  result.pythonPath = pythonCmd;

  // Check python version
  try {
    const version = await runCommand(pythonCmd, ["--version"]);
    result.pythonVersion = version.trim().replace("Python ", "");
    const [major, minor] = result.pythonVersion.split(".").map(Number);
    if (major < 3 || (major === 3 && minor < 10)) {
      result.error = `Python 3.10+ required, found ${result.pythonVersion}`;
      return result;
    }
  } catch {
    result.error = "Failed to get Python version";
    return result;
  }

  // Check mlx-lm
  try {
    const mlxVersion = await runCommand(pythonCmd, [
      "-c",
      "import mlx_lm; print(mlx_lm.__version__)",
    ]);
    result.mlxVersion = mlxVersion.trim();
  } catch {
    result.error =
      "mlx-lm not installed. Run: pip install mlx-lm";
    return result;
  }

  result.ok = true;
  return result;
}

async function findPython(): Promise<string | null> {
  // Try Homebrew's mlx-lm bundled Python first (has mlx_lm pre-installed)
  try {
    const prefix = (await runCommand("brew", ["--prefix", "mlx-lm"])).trim();
    const brewPython = path.join(prefix, "libexec", "bin", "python3");
    const version = await runCommand(brewPython, ["--version"]);
    if (version.includes("Python 3")) return brewPython;
  } catch {
    // Homebrew mlx-lm not installed, try standard paths
  }

  for (const cmd of ["python3", "python"]) {
    try {
      const version = await runCommand(cmd, ["--version"]);
      if (version.includes("Python 3")) return cmd;
    } catch {
      continue;
    }
  }
  return null;
}

// --- Model Download ---

export async function ensureBaseModel(
  pythonPath: string,
  model: string,
  onProgress: (status: string) => void
): Promise<boolean> {
  onProgress(`Checking if ${model} is cached...`);

  try {
    // Lightweight cache check — just see if snapshot_download resolves locally
    const check = await runCommand(pythonPath, [
      "-c",
      `from huggingface_hub import try_to_load_from_cache, model_info
import os
# Check if the config file is cached (proxy for full model)
result = try_to_load_from_cache("${model}", "config.json")
if result is not None and os.path.exists(result):
    print("cached")
else:
    print("missing")`,
    ]);
    if (check.trim() === "cached") {
      onProgress("Model already cached");
      return true;
    }
  } catch {
    // Cache check failed, proceed to download
  }

  onProgress(`Downloading ${model} (this may take a few minutes)...`);
  try {
    await runCommand(pythonPath, [
      "-c",
      `from huggingface_hub import snapshot_download; snapshot_download("${model}")`,
    ]);
    onProgress("Model downloaded");
    return true;
  } catch (err) {
    onProgress(`Download failed: ${err instanceof Error ? err.message : String(err)}`);
    return false;
  }
}

// --- Data Conversion ---

export function convertToSFTFormat(
  findings: DemoFinding[],
  corrections: Correction[],
  styleEdits: StyleEdit[],
  workflows: PrescribedWorkflow[]
): { train: object[]; valid: object[] } {
  const examples: object[] = [];

  // 1. Prediction corrections — train on ground truth
  for (const f of findings) {
    const gt = f.ground_truth;
    examples.push({
      messages: [
        { role: "system", content: ANALYSIS_SYSTEM_PROMPT },
        {
          role: "user",
          content: `Analyze this static analysis finding:\n\nFile: ${f.scan.file}\nLine: ${f.scan.line}\nType: ${f.scan.type}\nStatic severity: ${f.scan.severity}\nMatch: ${f.scan.match}\nContext: ${f.scan.context}`,
        },
        {
          role: "assistant",
          content: JSON.stringify({
            severity: gt.severity,
            is_false_positive: gt.is_false_positive,
            reasoning: gt.reasoning,
          }),
        },
      ],
    });
  }

  // 2. Workpaper style — train on reviewer-edited content
  for (const edit of styleEdits) {
    examples.push({
      messages: [
        { role: "system", content: WORKPAPER_SYSTEM_PROMPT },
        {
          role: "user",
          content: `Write the "${edit.sectionTitle}" section for this audit workpaper.`,
        },
        { role: "assistant", content: edit.edited },
      ],
    });
  }

  // 3. Investigation workflows — train on prescribed steps
  for (const wf of workflows) {
    examples.push({
      messages: [
        { role: "system", content: INVESTIGATION_SYSTEM_PROMPT },
        {
          role: "user",
          content: `Investigate this finding:\n\nFile: ${wf.file}\nType: ${wf.findingType}\nFinding ID: ${wf.findingId}`,
        },
        {
          role: "assistant",
          content: JSON.stringify({
            steps: wf.steps,
            disposition: wf.disposition,
          }),
        },
      ],
    });
  }

  // 80/20 train/valid split
  const shuffled = examples.sort(() => Math.random() - 0.5);
  const splitIdx = Math.max(1, Math.floor(shuffled.length * 0.8));

  return {
    train: shuffled.slice(0, splitIdx),
    valid: shuffled.slice(splitIdx),
  };
}

export function convertToDPOFormat(
  findings: DemoFinding[],
  corrections: Correction[],
  styleEdits: StyleEdit[]
): object[] {
  const pairs: object[] = [];

  // 1. Prediction preference pairs — where AI got it wrong
  for (const f of findings) {
    if (!f.ai_prediction) continue;
    const ai = f.ai_prediction;
    const gt = f.ground_truth;

    if (ai.severity !== gt.severity || ai.is_false_positive !== gt.is_false_positive) {
      const prompt = `${ANALYSIS_SYSTEM_PROMPT}\n\nAnalyze this static analysis finding:\n\nFile: ${f.scan.file}\nLine: ${f.scan.line}\nType: ${f.scan.type}\nStatic severity: ${f.scan.severity}\nMatch: ${f.scan.match}\nContext: ${f.scan.context}`;

      pairs.push({
        prompt,
        chosen: JSON.stringify({
          severity: gt.severity,
          is_false_positive: gt.is_false_positive,
          reasoning: gt.reasoning,
        }),
        rejected: JSON.stringify({
          severity: ai.severity,
          is_false_positive: ai.is_false_positive,
          reasoning: ai.reasoning,
        }),
      });
    }
  }

  // 2. Style preference pairs — reviewer edit vs base draft
  for (const edit of styleEdits) {
    const prompt = `${WORKPAPER_SYSTEM_PROMPT}\n\nWrite the "${edit.sectionTitle}" section for this audit workpaper.`;

    pairs.push({
      prompt,
      chosen: edit.edited,
      rejected: edit.original,
    });
  }

  return pairs;
}

export function writeSFTData(
  dataDir: string,
  data: { train: object[]; valid: object[] }
): void {
  fs.mkdirSync(dataDir, { recursive: true });

  const trainPath = path.join(dataDir, "train.jsonl");
  const validPath = path.join(dataDir, "valid.jsonl");

  fs.writeFileSync(
    trainPath,
    data.train.map((e) => JSON.stringify(e)).join("\n") + "\n",
    "utf-8"
  );
  fs.writeFileSync(
    validPath,
    data.valid.map((e) => JSON.stringify(e)).join("\n") + "\n",
    "utf-8"
  );
}

export function writeDPOData(dataPath: string, pairs: object[]): void {
  const dir = path.dirname(dataPath);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(
    dataPath,
    pairs.map((p) => JSON.stringify(p)).join("\n") + "\n",
    "utf-8"
  );
}

// --- Training ---

const PROGRESS_REGEX =
  /Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\d.e+-]+),\s+It\/sec\s+([\d.]+),.*?Peak mem\s+([\d.]+)\s+GB/;

export async function runSFTTraining(
  pythonPath: string,
  config: SFTConfig,
  onProgress: (p: TrainingProgress) => void
): Promise<string> {
  const adapterDir = path.join(config.outputDir, "sft-adapters");
  fs.mkdirSync(adapterDir, { recursive: true });

  const args = [
    "-m",
    "mlx_lm.lora",
    "--model",
    config.baseModel,
    "--data",
    config.dataDir,
    "--train",
    "--iters",
    String(config.iterations),
    "--batch-size",
    "1",
    "--num-layers",
    String(config.loraLayers),
    "--learning-rate",
    String(config.learningRate),
    "--adapter-path",
    adapterDir,
  ];

  await spawnWithProgress(pythonPath, args, config.iterations, "sft", onProgress);
  return adapterDir;
}

export async function runDPOTraining(
  pythonPath: string,
  scriptPath: string,
  config: DPOConfig,
  onProgress: (p: TrainingProgress) => void
): Promise<string> {
  const adapterDir = path.join(config.outputDir, "dpo-adapters");

  const args = [
    scriptPath,
    "--model",
    config.baseModel,
    "--adapter-path",
    config.sftAdapterPath,
    "--data",
    config.dataPath,
    "--output-adapter-path",
    adapterDir,
    "--iters",
    String(config.iterations),
    "--learning-rate",
    String(config.learningRate),
    "--beta",
    String(config.beta),
    "--num-layers",
    String(config.loraLayers),
  ];

  await spawnWithProgress(pythonPath, args, config.iterations, "dpo", onProgress);
  return adapterDir;
}

function spawnWithProgress(
  pythonPath: string,
  args: string[],
  totalIterations: number,
  phase: "sft" | "dpo",
  onProgress: (p: TrainingProgress) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn(pythonPath, args, {
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stderr = "";

    const handleLine = (line: string) => {
      const match = line.match(PROGRESS_REGEX);
      if (match) {
        onProgress({
          phase,
          iteration: parseInt(match[1]),
          totalIterations,
          loss: parseFloat(match[2]),
          learningRate: parseFloat(match[3]),
          iterPerSec: parseFloat(match[4]),
          peakMemGB: parseFloat(match[5]),
        });
      }
    };

    let stdoutBuffer = "";
    proc.stdout.on("data", (chunk: Buffer) => {
      stdoutBuffer += chunk.toString();
      const lines = stdoutBuffer.split("\n");
      stdoutBuffer = lines.pop() ?? "";
      for (const line of lines) {
        handleLine(line);
      }
    });

    proc.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    proc.on("close", (code) => {
      if (stdoutBuffer) handleLine(stdoutBuffer);
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Training failed (exit ${code}): ${stderr.slice(-500)}`));
      }
    });

    proc.on("error", reject);
  });
}

// --- MLX Server ---

export function startMLXServer(
  pythonPath: string,
  model: string,
  adapterPath: string | null,
  port: number
): ChildProcess {
  const args = [
    "-m",
    "mlx_lm.server",
    "--model",
    model,
    "--port",
    String(port),
  ];
  if (adapterPath) {
    args.push("--adapter-path", adapterPath);
  }
  const proc = spawn(pythonPath, args, {
    stdio: ["ignore", "pipe", "pipe"],
  });

  return proc;
}

export async function waitForMLXServer(
  port: number,
  timeoutMs = 30000
): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`http://localhost:${port}/v1/models`, {
        signal: AbortSignal.timeout(2000),
      });
      if (res.ok) return true;
    } catch {
      // Not ready yet
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  return false;
}

export function stopMLXServer(proc: ChildProcess): void {
  proc.kill("SIGTERM");
}

// --- Utilities ---

function runCommand(cmd: string, args: string[], timeoutMs = 300_000): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let killed = false;

    const timer = setTimeout(() => {
      killed = true;
      proc.kill("SIGTERM");
      reject(new Error(`Command timed out after ${timeoutMs / 1000}s`));
    }, timeoutMs);

    proc.stdout.on("data", (d: Buffer) => (stdout += d.toString()));
    proc.stderr.on("data", (d: Buffer) => (stderr += d.toString()));
    proc.on("close", (code) => {
      clearTimeout(timer);
      if (killed) return;
      if (code === 0) resolve(stdout);
      else reject(new Error(stderr || `Exit code ${code}`));
    });
    proc.on("error", (err) => {
      clearTimeout(timer);
      if (!killed) reject(err);
    });
  });
}
