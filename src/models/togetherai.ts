import Together from 'together-ai';

import { ModelProvider } from './base';
import { Usage } from '../types';
import { calculateTokenCost, OCR_SYSTEM_PROMPT } from './shared';

export class TogetherProvider extends ModelProvider {
  private client: Together;

  constructor(model: string) {
    super(model);

    const apiKey = process.env.TOGETHER_API_KEY;
    if (!apiKey) {
      throw new Error('Missing required Together API key');
    }

    this.client = new Together();
  }

  async ocr(imagePath: string): Promise<{
    text: string;
    imageBase64s?: string[];
    usage: Usage;
  }> {
    const start = performance.now();

    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: OCR_SYSTEM_PROMPT },
            { type: 'image_url', image_url: { url: imagePath } },
          ],
        },
      ],
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
