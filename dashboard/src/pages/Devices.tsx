import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Icon } from '../components/Icon'
import { DevicePill, Empty, Esp32Pill, useToast } from '../components/ui'
import { useAuth } from '../lib/auth'
import { useData } from '../lib/data'
import { deviceName, fmtAgo, fmtNumber, isOnline } from '../lib/format'
import type { Device } from '../lib/types'

function DeviceCard({ d }: { d: Device }) {
  const { now, events, updateDevice } = useData()
  const { profile } = useAuth()
  const toast = useToast()
  const nav = useNavigate()
  const [edit, setEdit] = useState(false)
  const [name, setName] = useState(d.name)
  const [loc, setLoc] = useState(d.location)
  const [busy, setBusy] = useState(false)
  const isAdmin = profile?.role === 'admin'
  const total = events.filter(e => e.device_id === d.id).length
  const online = isOnline(d, now)

  async function save() {
    setBusy(true)
    const err = await updateDevice(d.id, { name: name.trim(), location: loc.trim() })
    setBusy(false)
    toast(err ? 'Không lưu được: ' + err : 'Đã lưu thông tin thiết bị', !!err)
    if (!err) setEdit(false)
  }

  return (
    <div className="card dev-card">
      <div className="card-head">
        <h2 className="ellipsis">{deviceName(d)}</h2><DevicePill d={d} now={now} />
        {isAdmin && !edit && <div className="right"><button className="btn btn-sm btn-outline" onClick={() => setEdit(true)}><Icon name="edit" />Sửa</button></div>}
      </div>
      <div className="card-body">
        {edit ? (
          <>
            <div className="form-row">
              <div className="field" style={{ margin: 0 }}><label>Tên hiển thị</label><input className="input" value={name} onChange={e => setName(e.target.value)} maxLength={60} /></div>
              <div className="field" style={{ margin: 0 }}><label>Vị trí</label><input className="input" value={loc} onChange={e => setLoc(e.target.value)} maxLength={60} /></div>
            </div>
            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button className="btn btn-sm btn-outline" onClick={() => { setEdit(false); setName(d.name); setLoc(d.location) }}>Huỷ</button>
              <button className="btn btn-sm btn-primary" disabled={busy || !name.trim()} onClick={save}>Lưu</button>
            </div>
          </>
        ) : (
          <dl className="kv">
            <dt>Mã thiết bị</dt><dd>{d.id}</dd>
            <dt>Vị trí</dt><dd>{d.location || '–'}</dd>
            <dt>Heartbeat</dt><dd>{fmtAgo(d.last_seen, now)}</dd>
            <dt>Nhận diện</dt><dd>{online ? (d.state === 'unknown' ? 'Chưa chạy 4_realtime.py' : `${fmtNumber(d.fps ?? 0)} FPS · ${d.persons_count} người`) : '–'}</dd>
            <dt>Camera từ xa</dt><dd>{online && d.stream_url ? 'Sẵn sàng' : online ? 'Chưa có tunnel' : '–'}</dd>
            <dt>Hộp ESP32</dt><dd><Esp32Pill d={d} now={now} /></dd>
            <dt>Ca ngã đã ghi</dt><dd className="num">{total}</dd>
            <dt>Agent</dt><dd>{d.agent_version ? 'v' + d.agent_version : '–'}</dd>
          </dl>
        )}
        {!edit && <button className="btn btn-sm btn-outline" style={{ alignSelf: 'flex-start' }} onClick={() => nav(`/camera?device=${encodeURIComponent(d.id)}`)}><Icon name="cam" />Xem camera</button>}
      </div>
    </div>
  )
}

export function DevicesPage() {
  const { devices } = useData()
  const { profile } = useAuth()
  return (
    <>
      {profile?.role !== 'admin' && <div className="note" style={{ marginTop: 0 }}><Icon name="lock" /><span>Chỉ quản trị viên mới đổi được tên và vị trí thiết bị.</span></div>}
      {devices.length === 0 ? <div className="card"><Empty icon="dev">Chưa có thiết bị nào. Cài <b>dashboard_agent</b> lên Raspberry Pi, thiết bị sẽ tự xuất hiện ở đây.</Empty></div> : (
        <div className="dev-grid">{devices.map(d => <DeviceCard key={d.id} d={d} />)}</div>
      )}
    </>
  )
}
