import { useMemo, useState } from 'react'
import { Icon } from '../components/Icon'
import { BarChart, Heatmap } from '../components/charts'
import { Empty, StatusPill } from '../components/ui'
import { useDeviceFilter } from '../components/Layout'
import { useData } from '../lib/data'
import { addDays, avgResponse, countsPerDay, hourHeatmap, inRange, pctDelta, startOfDay } from '../lib/stats'
import { STATUS_LABEL, deviceName, fmtDate, fmtDuration, fmtTime, responseSeconds } from '../lib/format'
import type { FallEvent } from '../lib/types'

const RANGES = [7, 30, 90] as const

function exportCsv(rows: FallEvent[], name: (id: string) => string) {
  const head = ['thoi_diem_nga', 'thiet_bi', 'nguoi', 'bat_dong_giay', 'trang_thai', 'phan_hoi_giay', 'ghi_chu']
  const esc = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`
  const lines = rows.map(e => [
    new Date(e.fall_time || e.detected_at).toLocaleString('vi-VN'), name(e.device_id), e.person_id ?? '',
    e.duration_s ?? '', STATUS_LABEL[e.status], responseSeconds(e)?.toFixed(0) ?? '', e.note,
  ].map(esc).join(','))
  const blob = new Blob(['﻿' + [head.join(','), ...lines].join('\n')], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `ca-te-nga-${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
}

export function HistoryPage() {
  const { events, devices, deviceMap, now } = useData()
  const { deviceId } = useDeviceFilter()
  const [days, setDays] = useState<(typeof RANGES)[number]>(30)

  const s = useMemo(() => {
    const end = addDays(startOfDay(new Date(now)), 1)
    const from = addDays(end, -days), prevFrom = addDays(from, -days)
    const base = deviceId ? events.filter(e => e.device_id === deviceId) : events
    const cur = base.filter(e => inRange(e, from, end))
    const prev = base.filter(e => inRange(e, prevFrom, from))
    const durs = cur.map(e => e.duration_s).filter((x): x is number => x != null)
    const resp = cur.map(responseSeconds).filter((x): x is number => x != null)
    const fast = resp.filter(x => x <= 120).length
    const byDev = new Map<string, number>()
    cur.forEach(e => byDev.set(e.device_id, (byDev.get(e.device_id) || 0) + 1))
    return {
      cur, prev, from,
      delta: pctDelta(cur.length, prev.length),
      avgDur: durs.length ? durs.reduce((a, b) => a + b, 0) / durs.length : null,
      maxDur: durs.length ? Math.max(...durs) : null,
      fastPct: resp.length ? Math.round((fast / resp.length) * 100) : null,
      avgResp: avgResponse(cur),
      open: cur.filter(e => e.status !== 'resolved').length,
      byDev: [...byDev.entries()].sort((a, b) => b[1] - a[1]),
      chart: countsPerDay(base, days, new Date(now)),
      heat: hourHeatmap(cur),
    }
  }, [events, deviceId, days, now])

  const maxDev = Math.max(1, ...s.byDev.map(x => x[1]))
  const name = (id: string) => deviceName(deviceMap.get(id), id)

  return (
    <>
      <div className="row" style={{ justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <span className="muted" style={{ fontSize: 13 }}>Từ {fmtDate(s.from.toISOString())} đến {fmtDate(new Date(now).toISOString())}{deviceId ? ` · ${name(deviceId)}` : ` · ${devices.length} thiết bị`}</span>
        <div className="seg">{RANGES.map(r => <button key={r} className={days === r ? 'on' : ''} onClick={() => setDays(r)}>{r} ngày</button>)}</div>
      </div>

      <div className="kpis">
        <div className="card kpi"><div className="label">Tổng ca ngã · {days} ngày</div><div className="value num">{s.cur.length}</div>
          <div className="foot">{s.delta != null && <span className={`delta ${s.delta > 0 ? 'up-bad' : 'down-good'}`}>{s.delta > 0 ? '+' : ''}{s.delta}%</span>}so với {days} ngày trước ({s.prev.length})</div></div>
        <div className="card kpi"><div className="label">Thời gian bất động TB</div><div className="value num">{s.avgDur != null ? fmtDuration(s.avgDur) : '–'}</div>
          <div className="foot">{s.maxDur != null ? `dài nhất ${fmtDuration(s.maxDur)}` : 'chưa có dữ liệu'}</div></div>
        <div className="card kpi"><div className="label">Xác nhận trong 2 phút</div><div className="value num">{s.fastPct != null ? s.fastPct + '%' : '–'}</div>
          <div className="foot">{s.avgResp != null ? `phản hồi TB ${fmtDuration(Math.round(s.avgResp))}` : 'chưa có ca nào được xác nhận'}</div></div>
        <div className="card kpi"><div className="label">Chưa xử lý xong</div><div className="value num">{s.open}</div>
          <div className="foot">trong {days} ngày</div></div>
      </div>

      <div className="stats">
        <div className="card card-navy">
          <div className="card-head"><h2>Số ca ngã theo ngày</h2><span className="sub">{days} ngày gần nhất</span>
            <div className="right legend" style={{ color: '#93a7c1' }}><span><i style={{ background: '#7fb0ff' }} />Ca ngã</span><span><i style={{ background: '#93a7c1', height: 1 }} />Trung bình 7 ngày</span></div></div>
          <div className="card-body"><BarChart data={s.chart} dark height={330} showAvg labelEvery={days <= 7 ? 1 : days <= 30 ? 5 : 15} /></div>
        </div>
        <div className="card">
          <div className="card-head"><h2>Theo thiết bị</h2><span className="sub">{days} ngày</span></div>
          <div className="card-body">
            {s.byDev.length === 0 ? <div className="muted" style={{ fontSize: 13 }}>Chưa có ca ngã nào trong khoảng này.</div> : s.byDev.map(([id, n]) => (
              <div className="bar-row" key={id}><span className="ellipsis">{name(id)}</span><div className="bar-track"><div className="bar-fill" style={{ width: `${(n / maxDev) * 100}%` }} /></div><span className="num" style={{ textAlign: 'right' }}>{n}</span></div>
            ))}
            <div style={{ borderTop: '1px solid var(--line-2)', margin: '14px 0 12px' }} />
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>Khung giờ hay xảy ra</div>
            <Heatmap m={s.heat} />
            <div className="legend" style={{ marginTop: 10 }}><span>Ít</span><span><i style={{ background: '#c9d8ee', height: 8, width: 12 }} /></span><span><i style={{ background: '#7ea0cf', height: 8, width: 12 }} /></span><span><i style={{ background: '#1f4675', height: 8, width: 12 }} /></span><span>Nhiều</span></div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head"><h2>Nhật ký ca ngã</h2><span className="sub">{s.cur.length} ca</span>
          <div className="right"><button className="btn btn-sm btn-outline" disabled={!s.cur.length} onClick={() => exportCsv(s.cur, name)}><Icon name="dl" />Xuất CSV</button></div></div>
        {s.cur.length === 0 ? <Empty>Chưa có ca ngã nào trong {days} ngày qua</Empty> : (
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>Thời điểm</th><th>Thiết bị</th><th className="hide-sm">Người</th><th className="hide-sm">Bất động</th><th>Trạng thái</th><th className="hide-sm">Phản hồi sau</th><th className="hide-sm">Ghi chú</th></tr></thead>
              <tbody>
                {s.cur.slice(0, 100).map(e => (
                  <tr key={e.id}>
                    <td className="num">{fmtDate(e.fall_time || e.detected_at).slice(0, 5)} {fmtTime(e.fall_time || e.detected_at, false)}</td>
                    <td>{name(e.device_id)}</td>
                    <td className="hide-sm">#{e.person_id ?? '?'}</td>
                    <td className="num hide-sm">{fmtDuration(e.duration_s)}</td>
                    <td><StatusPill s={e.status} /></td>
                    <td className="num hide-sm">{responseSeconds(e) != null ? fmtDuration(Math.round(responseSeconds(e)!)) : <span className="faint">–</span>}</td>
                    <td className="hide-sm">{e.note || <span className="faint">–</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {s.cur.length > 100 && <div className="muted" style={{ padding: '10px 18px', fontSize: 12.5 }}>Hiển thị 100 / {s.cur.length} ca. Xuất CSV để xem đầy đủ.</div>}
          </div>
        )}
      </div>
    </>
  )
}
