import OpenAI from 'openai';
import sharp from 'sharp';

import { ModelProvider } from './base';
import { Usage } from '../types';
import { calculateTokenCost, OCR_SYSTEM_PROMPT } from './shared';

export class DashscopeProvider extends ModelProvider {
  private client: OpenAI;

  constructor(model: string) {
    super(model);

    const apiKey = process.env.DASHSCOPE_API_KEY;
    if (!apiKey) {
      throw new Error('Missing required HuggingFace API key');
    }

    this.client = new OpenAI({
      baseURL: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
      apiKey,
    });
  }

  async ocr(imagePath: string): Promise<{
    text: string;
    imageBase64s?: string[];
    usage: Usage;
  }> {
    const start = performance.now();

    // Fetch the image
    const imageResponse = await fetch(imagePath);
    const imageBuffer = await imageResponse.arrayBuffer();

    // compress the image
    const resizedImageBuffer = await sharp(Buffer.from(imageBuffer))
      .jpeg({ quality: 90 })
      .toBuffer();

    // Convert to base64
    const resizedImageBase64 = `data:image/jpeg;base64,${resizedImageBuffer.toString('base64')}`;

    const response = await this.client.chat.completions.create({
      model: 'qwen2.5-vl-32b-instruct',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: OCR_SYSTEM_PROMPT },
            {
              type: 'image_url',
              image_url: {
                url: resizedImageBase64,
              },
            },
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
