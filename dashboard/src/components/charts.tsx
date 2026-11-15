import { fmtNumber } from '../lib/format'

const DAY = (d: Date) => `${String(d.getDate()).padStart(2, '0')}/${String(d.getMonth() + 1).padStart(2, '0')}`

/** Bieu do cot don gian (SVG), cot cuoi = hom nay to mau do */
export function BarChart({ data, dark, height = 170, labelEvery = 1, showAvg }: {
  data: { day: Date; count: number }[]
  dark?: boolean
  height?: number
  labelEvery?: number
  showAvg?: boolean
}) {
  const W = 760, H = height, left = 36, right = 10, top = 14, bottom = 26
  const max = Math.max(3, ...data.map(d => d.count))
  const plotH = H - top - bottom
  const step = (W - left - right) / data.length
  const y = (v: number) => top + plotH - (v / max) * plotH
  const ticks = Array.from({ length: Math.min(max, 4) + 1 }, (_, i) => Math.round((i * max) / Math.min(max, 4)))
  const grid = dark ? 'rgba(255,255,255,.08)' : '#eef1f5'
  const txt = dark ? '#93a7c1' : '#8a96a6'
  const bar = dark ? '#7fb0ff' : '#1f4675'
  const today = dark ? '#ff6b6b' : '#dc2626'
  const avg = data.map((_, i) => {
    const s = data.slice(Math.max(0, i - 6), i + 1)
    return s.reduce((a, b) => a + b.count, 0) / s.length
  })
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }} fontFamily="Segoe UI, system-ui" role="img"
      aria-label={`Số ca ngã theo ngày, tổng ${data.reduce((a, b) => a + b.count, 0)}`}>
      {[...new Set(ticks)].map(t => (
        <g key={t}>
          <line x1={left} x2={W - right} y1={y(t)} y2={y(t)} stroke={grid} />
          <text x={left - 10} y={y(t) + 4} fontSize="11" fill={txt} textAnchor="end">{t}</text>
        </g>
      ))}
      {data.map((d, i) => d.count > 0 && (
        <rect key={i} x={left + i * step + step * 0.18} y={y(d.count)} width={step * 0.64} height={top + plotH - y(d.count)}
          fill={i === data.length - 1 ? today : bar}><title>{`${DAY(d.day)}: ${d.count} ca`}</title></rect>
      ))}
      {showAvg && <polyline fill="none" stroke={txt} strokeWidth="1.6" strokeDasharray="4 3"
        points={avg.map((v, i) => `${left + i * step + step / 2},${y(v)}`).join(' ')} />}
      {data.map((d, i) => (i % labelEvery === 0 || i === data.length - 1) && (
        <text key={i} x={left + i * step + step / 2} y={H - 8} fontSize="11" textAnchor="middle"
          fill={i === data.length - 1 ? (dark ? '#fff' : '#16202e') : txt} fontWeight={i === data.length - 1 ? 700 : 400}>
          {i === data.length - 1 ? 'Hôm nay' : DAY(d.day)}
        </text>
      ))}
    </svg>
  )
}

export function Heatmap({ m }: { m: number[][] }) {
  const max = Math.max(1, ...m.flat())
  const lvl = (v: number) => (v === 0 ? '' : v / max > 0.66 ? 'l3' : v / max > 0.33 ? 'l2' : 'l1')
  const days = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN']
  return (
    <div className="hm">
      <span />
      {Array.from({ length: 24 }, (_, h) => <span key={h} style={{ textAlign: 'center' }}>{h % 6 === 0 ? `${h}h` : ''}</span>)}
      {days.map((d, di) => [
        <span key={d}>{d}</span>,
        ...m[di].map((v, h) => <i key={`${d}${h}`} className={lvl(v)} title={`${d} ${h}h: ${v} ca`} />),
      ])}
    </div>
  )
}

export function ProbBars({ probs }: { probs: number[] | null }) {
  const rows: [string, string][] = [['Bình thường', '#15803d'], ['Ngồi', '#b45309'], ['Nằm', '#b45309'], ['Té ngã', '#dc2626']]
  return (
    <div className="probs">
      {rows.map(([label, color], i) => {
        const p = probs ? probs[i] ?? 0 : 0
        const strong = probs && i === probs.indexOf(Math.max(...probs))
        return (
          <div className="prow" key={label}>
            <span>{strong ? <b>{label}</b> : label}</span>
            <div className="track"><div className="fill" style={{ width: `${Math.round(p * 100)}%`, background: color }} /></div>
            <span className="num" style={{ textAlign: 'right' }}>{probs ? (strong ? <b>{fmtNumber(p * 100, 0)}%</b> : <span className="muted">{fmtNumber(p * 100, 0)}%</span>) : <span className="faint">–</span>}</span>
          </div>
        )
      })}
    </div>
  )
}
