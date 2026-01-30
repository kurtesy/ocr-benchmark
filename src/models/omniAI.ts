import axios from 'axios';
import FormData from 'form-data';

import { JsonSchema } from '../types';
import { ModelProvider } from './base';

// https://getomni.ai/pricing
// 1 cent per page
const COST_PER_PAGE = 0.01;

interface ExtractResponse {
  ocr: {
    pages: Array<{
      page: number;
      content: string;
    }>;
    inputTokens: number;
    outputTokens: number;
  };
  extracted?: Record<string, any>; // Only present when schema is provided
}

export const sendExtractRequest = async (
  imageUrl: string,
  schema?: JsonSchema,
): Promise<ExtractResponse> => {
  const apiKey = process.env.OMNIAI_API_KEY;
  if (!apiKey) {
    throw new Error('Missing OMNIAI_API_KEY in .env');
  }

  const formData = new FormData();
  formData.append('url', imageUrl);

  // Add optional parameters if provided
  if (schema) {
    formData.append('schema', JSON.stringify(schema));
  }

  try {
    const response = await axios.post(
      `${process.env.OMNIAI_API_URL}/extract/sync`,
      formData,
      {
        headers: {
          'x-api-key': apiKey,
          ...formData.getHeaders(),
        },
      },
    );

    return response.data.result;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(
        `Failed to extract from image: ${JSON.stringify(error.response?.data) || JSON.stringify(error.message)}`,
      );
    }
    throw error;
  }
};

export class OmniAIProvider extends ModelProvider {
  constructor(model: string) {
    super(model);
  }

  async ocr(imagePath: string) {
    const start = performance.now();
    const response = await sendExtractRequest(imagePath);
    const end = performance.now();

    const text = response.ocr.pages.map((page) => page.content).join('\n');
    const inputTokens = response.ocr.inputTokens;
    const outputTokens = response.ocr.outputTokens;

    return {
      text,
      usage: {
        duration: end - start,
        inputTokens,
        outputTokens,
        totalTokens: inputTokens + outputTokens,
        totalCost: COST_PER_PAGE,
      },
    };
  }

  async extractFromImage(imagePath: string, schema?: JsonSchema) {
    const start = performance.now();
    const response = await sendExtractRequest(imagePath, schema);
    const end = performance.now();

    const inputToken = response.ocr.inputTokens;
    const outputToken = response.ocr.outputTokens;

    return {
      json: response.extracted || {},
      usage: {
        duration: end - start,
        inputTokens: inputToken,
        outputTokens: outputToken,
        totalTokens: inputToken + outputToken,
        totalCost: 0, // TODO: extraction cost is included in the OCR cost, 1 cent per page
      },
    };
  }
}
