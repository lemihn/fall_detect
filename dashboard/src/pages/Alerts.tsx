import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Icon } from '../components/Icon'
import { Empty, Snapshot, StatusPill, Thumb, useToast } from '../components/ui'
import { useDeviceFilter } from '../components/Layout'
import { useData } from '../lib/data'
import { deviceName, fmtDayLabel, fmtDuration, fmtTime, shortId } from '../lib/format'
import type { EventStatus, FallEvent } from '../lib/types'

type Filter = 'all' | EventStatus

function Detail({ e, onClose }: { e: FallEvent; onClose: () => void }) {
  const { deviceMap, updateEventStatus, updateEventNote } = useData()
  const nav = useNavigate()
  const toast = useToast()
  const [note, setNote] = useState(e.note)
  const [busy, setBusy] = useState(false)
  useEffect(() => setNote(e.note), [e.id, e.note])
  const d = deviceMap.get(e.device_id)

  async function setStatus(s: EventStatus) {
    setBusy(true)
    if (note !== e.note) await updateEventNote(e.id, note)
    const err = await updateEventStatus(e.id, s)
    setBusy(false)
    toast(err ? 'Lỗi: ' + err : s === 'resolved' ? 'Đã đánh dấu đã xử lý' : s === 'acknowledged' ? 'Đã xác nhận ca ngã' : 'Đã mở lại ca ngã', !!err)
  }

  return (
    <>
      <div className="card-head">
        <h2>Chi tiết ca ngã</h2><StatusPill s={e.status} />
        <div className="right"><span className="faint" style={{ fontSize: 12 }}>{shortId(e.id)}</span>
          <button className="icon-btn" style={{ width: 28, height: 28 }} onClick={onClose} aria-label="Đóng"><Icon name="x" /></button></div>
      </div>
      <Snapshot path={e.snapshot_path} label="Ảnh lúc cảnh báo" />
      <div className="card-body">
        <dl className="kv">
          <dt>Thiết bị</dt><dd>{deviceName(d, e.device_id)}{d?.location ? ` · ${d.location}` : ''}</dd>
          <dt>Người</dt><dd>#{e.person_id ?? '?'}</dd>
          <dt>Bắt đầu ngã</dt><dd>{fmtTime(e.fall_time || e.detected_at)} · {fmtDayLabel(e.fall_time || e.detected_at)}</dd>
          <dt>Bất động khi báo</dt><dd>{fmtDuration(e.duration_s)}</dd>
        </dl>
      </div>
      <ul className="timeline">
        {e.fall_time && <li><span className="tm">{fmtTime(e.fall_time)}</span><span className="dot red" /><span>Phát hiện ngã, bật còi và đèn tại chỗ</span></li>}
        <li><span className="tm">{fmtTime(e.detected_at)}</span><span className="dot amber" /><span>Bất động quá lâu, gửi cảnh báo Telegram và dashboard</span></li>
        {e.acknowledged_at
          ? <li><span className="tm">{fmtTime(e.acknowledged_at)}</span><span className="dot blue" /><span>Đã xác nhận</span></li>
          : <li><span className="tm">–</span><span className="dot" /><span className="muted">Chờ xác nhận…</span></li>}
        {e.resolved_at && <li><span className="tm">{fmtTime(e.resolved_at)}</span><span className="dot green" /><span>Đã xử lý xong</span></li>}
      </ul>
      <div style={{ padding: '4px 18px 12px' }}>
        <textarea className="note-input" placeholder="Thêm ghi chú (ví dụ: đã gọi người nhà kiểm tra)…" value={note} onChange={ev => setNote(ev.target.value)}
          onBlur={async () => { if (note !== e.note) { const err = await updateEventNote(e.id, note); toast(err ? 'Không lưu được ghi chú' : 'Đã lưu ghi chú', !!err) } }} />
      </div>
      <div className="detail-actions">
        <button className="btn btn-outline" onClick={() => nav(`/camera?device=${encodeURIComponent(e.device_id)}`)}><Icon name="cam" />Mở camera</button>
        {e.status === 'new' && <button className="btn btn-primary" disabled={busy} onClick={() => setStatus('acknowledged')}><Icon name="check" />Xác nhận đã biết</button>}
        {e.status === 'acknowledged' && <button className="btn btn-primary" disabled={busy} onClick={() => setStatus('resolved')}><Icon name="check" />Đã xử lý xong</button>}
        {e.status === 'resolved' && <button className="btn btn-outline" disabled={busy} onClick={() => setStatus('new')}>Mở lại</button>}
      </div>
    </>
  )
}

export function AlertsPage() {
  const { events, deviceMap } = useData()
  const { deviceId } = useDeviceFilter()
  const [params, setParams] = useSearchParams()
  const [filter, setFilter] = useState<Filter>('all')
  const [q, setQ] = useState('')

  const base = deviceId ? events.filter(e => e.device_id === deviceId) : events
  const counts = useMemo(() => ({
    all: base.length,
    new: base.filter(e => e.status === 'new').length,
    acknowledged: base.filter(e => e.status === 'acknowledged').length,
    resolved: base.filter(e => e.status === 'resolved').length,
  }), [base])

  const list = useMemo(() => {
    const s = q.trim().toLowerCase()
    return base.filter(e => (filter === 'all' || e.status === filter) && (!s ||
      deviceName(deviceMap.get(e.device_id), e.device_id).toLowerCase().includes(s) ||
      e.device_id.toLowerCase().includes(s) || e.note.toLowerCase().includes(s)))
  }, [base, filter, q, deviceMap])

  const selId = params.get('id')
  const selected = (selId && events.find(e => e.id === selId)) || null
  const shown = selected || list[0] || null
  const select = (id: string | null) => setParams(id ? { id } : {})

  const tabs: [Filter, string][] = [['all', 'Tất cả'], ['new', 'Mới'], ['acknowledged', 'Đã xác nhận'], ['resolved', 'Đã xử lý']]

  return (
    <div className="alerts-layout">
      <div className="card">
        <div className="filters">
          <div className="seg">{tabs.map(([k, l]) => <button key={k} className={filter === k ? 'on' : ''} onClick={() => setFilter(k)}>{l} · {counts[k]}</button>)}</div>
          <label className="search" style={{ marginLeft: 'auto' }}><Icon name="search" /><input placeholder="Tìm theo thiết bị, ghi chú…" value={q} onChange={e => setQ(e.target.value)} /></label>
        </div>
        {list.length === 0 ? <Empty icon="check">{base.length ? 'Không có ca nào khớp bộ lọc' : 'Chưa có ca té ngã nào được ghi nhận'}</Empty> : (
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>Ảnh</th><th>Thời điểm ngã</th><th>Thiết bị</th><th className="hide-sm">Người</th><th className="hide-sm">Bất động</th><th>Trạng thái</th></tr></thead>
              <tbody>
                {list.map(e => (
                  <tr key={e.id} className={`clickable${shown?.id === e.id ? ' sel' : ''}`} onClick={() => select(e.id)}>
                    <td><Thumb path={e.snapshot_path} /></td>
                    <td><b>{fmtTime(e.fall_time || e.detected_at)}</b><div className="muted" style={{ fontSize: 12 }}>{fmtDayLabel(e.fall_time || e.detected_at)}</div></td>
                    <td>{deviceName(deviceMap.get(e.device_id), e.device_id)}</td>
                    <td className="num hide-sm">#{e.person_id ?? '?'}</td>
                    <td className="num hide-sm">{fmtDuration(e.duration_s)}</td>
                    <td><StatusPill s={e.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      <div className={`sheet-backdrop${selected ? ' open' : ''}`} onClick={() => select(null)} />
      <div className={`card detail${selected ? ' open' : ''}`}>
        {shown ? <Detail e={shown} onClose={() => select(null)} /> : <Empty>Chọn một ca ngã để xem chi tiết</Empty>}
      </div>
    </div>
  )
}
