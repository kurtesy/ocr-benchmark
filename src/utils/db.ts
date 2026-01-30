import { PrismaClient } from '@prisma/client';
import { Result } from '../types';

const prisma = new PrismaClient();

export async function createBenchmarkRun(
  timestamp: string,
  modelsConfig: any,
  totalDocuments: number,
) {
  return prisma.benchmarkRun.create({
    data: {
      timestamp,
      status: 'running',
      modelsConfig: { models: modelsConfig },
      totalDocuments,
    },
  });
}

export async function saveResult(runId: string, result: Result) {
  return prisma.benchmarkResult.create({
    data: {
      benchmarkRunId: runId,
      fileUrl: result.fileUrl,
      metadata: result.metadata as any,
      ocrModel: result.ocrModel,
      extractionModel: result.extractionModel || '',
      jsonSchema: result.jsonSchema as any,
      directImageExtraction: result.directImageExtraction || false,
      trueMarkdown: result.trueMarkdown,
      trueJson: result.trueJson,
      predictedMarkdown: result.predictedMarkdown,
      predictedJson: result.predictedJson,
      levenshteinDistance: result.levenshteinDistance,
      jsonAccuracy: result.jsonAccuracy,
      jsonDiff: result.jsonDiff,
      fullJsonDiff: result.fullJsonDiff,
      jsonDiffStats: result.jsonDiffStats,
      jsonAccuracyResult: result.jsonAccuracyResult as any,
      usage: result.usage as any,
      error: JSON.stringify(result.error),
    },
  });
}

export async function completeBenchmarkRun(runId: string, error?: string) {
  return prisma.benchmarkRun.update({
    where: { id: runId },
    data: {
      status: error ? 'failed' : 'completed',
      completedAt: new Date(),
      error,
    },
  });
}

// Clean up function
export async function disconnect() {
  await prisma.$disconnect();
}
