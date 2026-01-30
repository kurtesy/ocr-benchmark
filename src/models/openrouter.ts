import OpenAI from 'openai';
import { ChatCompletionMessageParam } from 'openai/resources/chat/completions';

import { ModelProvider } from './base';
import { Usage } from '../types';
import { calculateTokenCost, OCR_SYSTEM_PROMPT } from './shared';

export class OpenRouterProvider extends ModelProvider {
  private client: OpenAI;

  constructor(model: string) {
    super(model);

    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
      throw new Error('Missing required OpenRouter API key');
    }

    this.client = new OpenAI({
      baseURL: 'https://openrouter.ai/api/v1',
      apiKey,
      defaultHeaders: {
        'HTTP-Referer': process.env.SITE_URL || 'https://github.com/omni-ai/benchmark',
        'X-Title': 'OmniAI OCR Benchmark',
      },
    });
  }

  async ocr(imagePath: string): Promise<{
    text: string;
    imageBase64s?: string[];
    usage: Usage;
  }> {
    const start = performance.now();

    const messages: ChatCompletionMessageParam[] = [
      {
        role: 'user',
        content: [
          { type: 'text', text: OCR_SYSTEM_PROMPT },
          { type: 'image_url', image_url: { url: imagePath } },
        ],
      },
    ];

    const response = await this.client.chat.completions.create({
      model: this.model,
      messages,
    });

    const end = performance.now();

    const inputTokens = response.usage?.prompt_tokens || 0;
    const outputTokens = response.usage?.completion_tokens || 0;
    const inputCost = calculateTokenCost(this.model, 'input', inputTokens);
    const outputCost = calculateTokenCost(this.model, 'output', outputTokens);

    return {
      text: response.choices[0].message.content || '',
      usage: {
        duration: end - start,
        inputTokens,
        outputTokens,
        totalTokens: inputTokens + outputTokens,
        inputCost,
        outputCost,
        totalCost: inputCost + outputCost,
      },
    };
  }
}
