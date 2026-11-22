import { create } from "zustand";
import { persist } from "zustand/middleware";
import { getGpu } from "@/lib/amp/gpus";
import { interpretInstruction, runOptimization } from "@/lib/amp/run";
import { getSample, SAMPLES } from "@/lib/amp/samples";
import type { ModuleId, Objective, PrecisionPolicy, RunResult } from "@/lib/amp/types";

const starter = SAMPLES[0];
const starterGpu = getGpu("rtx-2050");

function makeExample(): RunResult {
  return runOptimization({
    code: starter.code,
    gpu: starterGpu,
    objective: "latency",
    precisionPolicy: "auto",
    memoryBudgetGb: starterGpu.vramGb,
    instruction: starter.instruction,
    sampleId: starter.id,
    id: "example-transformer",
    createdAt: 0,
    example: true,
  });
}

const exampleRun = makeExample();

type AmpState = {
  module: ModuleId;
  gpuId: string;
  code: string;
  sampleId: string | null;
  instruction: string;
  objective: Objective;
  precisionPolicy: PrecisionPolicy;
  memoryBudgetGb: number;
  agentBudget: number;
  lastRun: RunResult;
  history: RunResult[];
  notes: string[];
  llmModelId: string;
  promptTokens: number;
  generatedTokens: number;
  llmBatch: number;
  continuous: boolean;
  speculative: boolean;
  parallelCount: number;
  parallelTraining: boolean;
  parallelBatch: number;
  parallelSeq: number;
  micros: number;
  setModule: (module: ModuleId) => void;
  setGpu: (id: string) => void;
  setCode: (code: string) => void;
  setInstruction: (instruction: string) => void;
  setObjective: (objective: Objective) => void;
  setPolicy: (precisionPolicy: PrecisionPolicy) => void;
  setBudget: (memoryBudgetGb: number) => void;
  setAgentBudget: (agentBudget: number) => void;
  loadSample: (id: string) => void;
  run: (agent?: boolean) => void;
  openRun: (id: string) => void;
  forgetRun: (id: string) => void;
  setLlm: (patch: Partial<Pick<AmpState, "llmModelId" | "promptTokens" | "generatedTokens" | "llmBatch" | "continuous" | "speculative">>) => void;
  setParallel: (
    patch: Partial<Pick<AmpState, "parallelCount" | "parallelTraining" | "parallelBatch" | "parallelSeq" | "micros">>,
  ) => void;
};

export const useAmp = create<AmpState>()(
  persist(
    (set, get) => ({
      module: "overview",
      gpuId: starterGpu.id,
      code: starter.code,
      sampleId: starter.id,
      instruction: starter.instruction,
      objective: "latency",
      precisionPolicy: "auto",
      memoryBudgetGb: starterGpu.vramGb,
      agentBudget: 6,
      lastRun: exampleRun,
      history: [exampleRun],
      notes: [],
      llmModelId: "llama3-8b",
      promptTokens: 2048,
      generatedTokens: 256,
      llmBatch: 1,
      continuous: true,
      speculative: false,
      parallelCount: 4,
      parallelTraining: false,
      parallelBatch: 8,
      parallelSeq: 2048,
      micros: 8,
      setModule: (module) => set({ module }),
      setGpu: (id) =>
        set((state) => {
          const prev = getGpu(state.gpuId);
          const next = getGpu(id);
          const untouched = Math.abs(state.memoryBudgetGb - prev.vramGb) < 0.05;
          return {
            gpuId: next.id,
            memoryBudgetGb: untouched ? next.vramGb : Math.min(state.memoryBudgetGb, next.vramGb),
          };
        }),
      setCode: (code) => set({ code, sampleId: null }),
      setInstruction: (instruction) => set({ instruction }),
      setObjective: (objective) => set({ objective }),
      setPolicy: (precisionPolicy) => set({ precisionPolicy }),
      setBudget: (memoryBudgetGb) => set({ memoryBudgetGb }),
      setAgentBudget: (agentBudget) => set({ agentBudget }),
      loadSample: (id) => {
        const sample = getSample(id);
        set({ sampleId: sample.id, code: sample.code, instruction: sample.instruction, module: "optimizer" });
      },
      run: (agent) => {
        const state = get();
        const read = interpretInstruction(state.instruction, state);
        const gpu = getGpu(read.gpuId);
        const budget = Math.min(read.memoryBudgetGb, gpu.vramGb);
        const result = runOptimization({
          code: state.code,
          gpu,
          objective: read.objective,
          precisionPolicy: read.precisionPolicy,
          memoryBudgetGb: budget,
          instruction: state.instruction,
          sampleId: state.sampleId,
          id: `run-${Date.now().toString(36)}`,
          createdAt: Date.now(),
          agent,
          budget: state.agentBudget,
        });
        set({
          gpuId: gpu.id,
          objective: read.objective,
          precisionPolicy: read.precisionPolicy,
          memoryBudgetGb: budget,
          notes: read.notes,
          lastRun: result,
          history: [result, ...state.history.filter((item) => item.id !== result.id)].slice(0, 24),
        });
      },
      openRun: (id) => {
        const found = get().history.find((item) => item.id === id);
        if (!found) return;
        set({
          lastRun: found,
          code: found.code,
          instruction: found.instruction,
          gpuId: found.gpuId,
          sampleId: found.sampleId,
          objective: found.objective,
          precisionPolicy: found.precisionPolicy,
          memoryBudgetGb: found.memoryBudgetGb,
          module: found.agentLog ? "agent" : "optimizer",
        });
      },
      forgetRun: (id) =>
        set((state) => {
          const history = state.history.filter((item) => item.id !== id);
          const lastRun = state.lastRun.id === id ? (history[0] ?? state.lastRun) : state.lastRun;
          return { history, lastRun };
        }),
      setLlm: (patch) => set(patch),
      setParallel: (patch) => set(patch),
    }),
    {
      name: "amp-console-v1",
      skipHydration: true,
      partialize: (state) => ({
        module: state.module,
        gpuId: state.gpuId,
        code: state.code,
        sampleId: state.sampleId,
        instruction: state.instruction,
        objective: state.objective,
        precisionPolicy: state.precisionPolicy,
        memoryBudgetGb: state.memoryBudgetGb,
        agentBudget: state.agentBudget,
        lastRun: state.lastRun,
        history: state.history,
        llmModelId: state.llmModelId,
        promptTokens: state.promptTokens,
        generatedTokens: state.generatedTokens,
        llmBatch: state.llmBatch,
        continuous: state.continuous,
        speculative: state.speculative,
        parallelCount: state.parallelCount,
        parallelTraining: state.parallelTraining,
        parallelBatch: state.parallelBatch,
        parallelSeq: state.parallelSeq,
        micros: state.micros,
      }),
    },
  ),
);

export function hydrateAmp() {
  void useAmp.persist.rehydrate();
}
