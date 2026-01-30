import {
  calculateJsonAccuracy,
  countTotalFields,
  countChanges,
} from '../../src/evaluation/json';

describe('countTotalFields', () => {
  it('counts fields in nested objects including array elements', () => {
    const obj = { a: 1, b: { c: 2, d: [3, { e: 4 }] } };
    expect(countTotalFields(obj)).toBe(4);
  });

  it('counts array elements as individual fields', () => {
    const obj = { a: [1, 2, 3], b: 'test', c: true };
    expect(countTotalFields(obj)).toBe(5);
  });

  it('counts nested objects within arrays', () => {
    const obj = { a: [{ b: 1 }, { c: 2 }], d: 'test', e: true };
    expect(countTotalFields(obj)).toBe(4);
  });

  it('includes null values in field count', () => {
    const obj = { a: null, b: { c: null }, d: 'test' };
    expect(countTotalFields(obj)).toBe(3);
  });

  it('excludes fields with __diff metadata suffixes', () => {
    const obj = {
      a: 1,
      b__deleted: true,
      c__added: 'test',
      d: { e: 2 },
    };
    expect(countTotalFields(obj)).toBe(2);
  });
});

describe('calculateJsonAccuracy', () => {
  it('returns 0.5 when half of the fields match', () => {
    const actual = { a: 1, b: 2 };
    const predicted = { a: 1, b: 3 };
    const result = calculateJsonAccuracy(actual, predicted);
    expect(result.score).toBe(0.5);
  });

  it('handles nested objects in accuracy calculation', () => {
    const actual = { a: 1, b: { c: 2, d: 4, e: 4 } };
    const predicted = { a: 1, b: { c: 2, d: 4, e: 5 } };
    const result = calculateJsonAccuracy(actual, predicted);
    expect(result.score).toBe(0.75);
  });

  it('calculates accuracy for nested arrays and objects', () => {
    const actual = { a: 1, b: [{ c: 2, d: 4, e: 4, f: [2, 9] }] };
    const predicted = { a: 1, b: [{ c: 2, d: 4, e: 5, f: [2, 3] }] };
    const result = calculateJsonAccuracy(actual, predicted);
    expect(result.score).toBe(0.5);
  });

  it('considers array elements matching regardless of order', () => {
    const actual = {
      a: 1,
      b: [
        { c: 1, d: 2 },
        { c: 3, d: 4 },
      ],
    };
    const predicted = {
      a: 1,
      b: [
        { c: 3, d: 4 },
        { c: 1, d: 2 },
      ],
    };
    const result = calculateJsonAccuracy(actual, predicted);
    expect(result.score).toBe(1);
  });

  it('counts all array elements as unmatched when predicted array is null', () => {
    const actual = { a: 1, b: [1, 2, 3] };
    const predicted = { a: 1, b: null };
    const result = calculateJsonAccuracy(actual, predicted);
    expect(result.score).toBe(1 / 4);
  });

  it('counts all nested array objects as unmatched when predicted is null', () => {
    const actual = { a: 1, b: [{ c: 1, d: 1 }, { c: 2 }, { c: 3, e: 4 }] };
    const predicted = { a: 1, b: null };
    const result = calculateJsonAccuracy(actual, predicted);
    expect(result.score).toBe(Number((1 / 6).toFixed(4)));
  });

  it('considers null fields in predicted object as partial matches', () => {
    const actual = { a: 1, b: { c: 1, d: { e: 1, f: 2 } } };
    const predicted = { a: 1, b: { c: 1, d: null } };
    const result = calculateJsonAccuracy(actual, predicted);
    expect(result.score).toBe(0.5);
  });

  describe('null value comparisons', () => {
    it('handles actual null to predicted value comparison', () => {
      const actual = { a: [{ b: 1, c: null }] };
      const predicted = { a: [{ b: 1, c: 2 }] };
      const result = calculateJsonAccuracy(actual, predicted);
      expect(result.score).toBe(0.5);
    });

    it('handles actual null to predicted object comparison', () => {
      const actual = { a: [{ b: 1, c: null, f: 4 }] };
      const predicted = { a: [{ b: 1, c: { d: 2 }, f: 4 }] };
      const result = calculateJsonAccuracy(actual, predicted);
      expect(result.score).toBe(0.6667);
    });

    it('handles actual null to predicted complex object comparison', () => {
      const actual = { a: [{ b: 1, c: null, f: 4 }] };
      const predicted = { a: [{ b: 1, c: { d: 2, e: 3 }, f: 4 }] };
      const result = calculateJsonAccuracy(actual, predicted);
      expect(result.score).toBe(0.3333);
    });

    it('handles actual null to predicted array comparison', () => {
      const actual = { a: [{ b: 1, c: null, f: 4 }] };
      const predicted = { a: [{ b: 1, c: [3], f: 4 }] };
      const result = calculateJsonAccuracy(actual, predicted);
      expect(result.score).toBe(0.6667);
    });

    it('handles actual value to predicted null comparison', () => {
      const actual = { a: [{ b: 1, c: 2 }] };
      const predicted = { a: [{ b: 1, c: null }] };
      const result = calculateJsonAccuracy(actual, predicted);
      expect(result.score).toBe(0.5);
    });

    it('handles actual object to predicted null comparison', () => {
      const actual = { a: [{ b: 1, c: { d: 2 } }] };
      const predicted = { a: [{ b: 1, c: null }] };
      const result = calculateJsonAccuracy(actual, predicted);
      expect(result.score).toBe(0.5);
    });

    it('handles actual complex object to predicted null comparison', () => {
      const actual = { a: [{ b: 1, c: { d: 2, e: 3 } }] };
      const predicted = { a: [{ b: 1, c: null }] };
      const result = calculateJsonAccuracy(actual, predicted);
      expect(result.score).toBe(0.3333);
    });

    it('handles actual array to predicted null comparison', () => {
      const actual = { a: [{ b: 1, c: [3, 2] }] };
      const predicted = { a: [{ b: 1, c: null }] };
      const result = calculateJsonAccuracy(actual, predicted);
      expect(result.score).toBe(0.3333);
    });
  });
});
