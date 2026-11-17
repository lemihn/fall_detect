import type { FallEvent } from './types'
import { responseSeconds } from './format'

export function startOfDay(d: Date): Date {
  const x = new Date(d)
  x.setHours(0, 0, 0, 0)
  return x
}

export function addDays(d: Date, n: number): Date {
  const x = new Date(d)
  x.setDate(x.getDate() + n)
  return x
}

/** So ca nga moi ngay trong `days` ngay gan nhat (phan tu cuoi = hom nay) */
export function countsPerDay(events: FallEvent[], days: number, now = new Date()): { day: Date; count: number }[] {
  const today = startOfDay(now)
  const out = Array.from({ length: days }, (_, i) => ({ day: addDays(today, i - days + 1), count: 0 }))
  const first = out[0].day.getTime()
  for (const e of events) {
    const t = new Date(e.fall_time || e.detected_at)
    const idx = Math.floor((startOfDay(t).getTime() - first) / 86400_000)
    if (idx >= 0 && idx < days) out[idx].count++
  }
  return out
}

export function inRange(e: FallEvent, from: Date, to: Date): boolean {
  const t = new Date(e.fall_time || e.detected_at).getTime()
  return t >= from.getTime() && t < to.getTime()
}

export function avgResponse(events: FallEvent[]): number | null {
  const xs = events.map(responseSeconds).filter((x): x is number => x != null && x >= 0)
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null
}

/** Ma tran 7 x 24: [thu trong tuan (T2..CN)][gio] = so ca */
export function hourHeatmap(events: FallEvent[]): number[][] {
  const m = Array.from({ length: 7 }, () => Array(24).fill(0) as number[])
  for (const e of events) {
    const t = new Date(e.fall_time || e.detected_at)
    m[(t.getDay() + 6) % 7][t.getHours()]++
  }
  return m
}

export function pctDelta(cur: number, prev: number): number | null {
  if (!prev) return null
  return Math.round(((cur - prev) / prev) * 100)
}
