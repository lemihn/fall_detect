import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Icon } from '../components/Icon'
import { ProbBars } from '../components/charts'
import { DevicePill, Empty, Esp32Pill, StreamView, Thumb, useToast } from '../components/ui'
import { useData } from '../lib/data'
import { accessToken, hasStream, resetAlarm, streamBase, switchCamera, usePiStatus } from '../lib/gateway'
import { deviceName, deviceView, fmtAgo, fmtDuration, fmtNumber, fmtTime, isOnline } from '../lib/format'
import type { Device } from '../lib/types'

/** Anh xem truoc cua thiet bi khac: lay /snapshot moi 10 giay thay vi mo them luong video */
function MiniPreview({ d, now }: { d: Device; now: number }) {
  const [src, setSrc] = useState<string | null>(null)
  const ok = hasStream(d, now)
  const base = streamBase(d)
  useEffect(() => {
    if (!ok) { setSrc(null); return }
    let alive = true
    const load = async () => {
      const t = await accessToken()
      if (alive && t) setSrc(`${base}/snapshot?token=${encodeURIComponent(t)}&t=${Date.now()}`)
    }
    load()
    const id = setInterval(load, 10_000)
    return () => { alive = false; clearInterval(id) }
  }, [ok, base])
  return (
    <div className="cam">
      {src ? <img className="stream" src={src} alt={deviceName(d)} onError={() => setSrc(null)} /> : (
        <div className="placeholder"><div>{!isOnline(d, now) ? <><Icon name="wifioff" />Camera offline<br /><span style={{ fontSize: 12 }}>Mất kết nối {fmtAgo(d.last_seen, now)}</span></> : 'Chưa có hình'}</div></div>
      )}
      {src && <span className="tag live">ẢNH 10 GIÂY</span>}
    </div>
  )
}

export function CameraPage() {
  const { devices, events, now, deviceMap, updateEventStatus } = useData()
  const [params, setParams] = useSearchParams()
  const nav = useNavigate()
  const toast = useToast()
  const [busy, setBusy] = useState(false)

  const selected: Device | undefined = useMemo(() => {
    const id = params.get('device')
    return (id && deviceMap.get(id)) || devices.find(d => deviceView(d, now) === 'fall') || devices.find(d => isOnline(d, now)) || devices[0]
  }, [params, deviceMap, devices, now])

  const st = usePiStatus(selected, now)
  const open = events.filter(e => e.device_id === selected?.id && e.status === 'new')
  const fallen = st?.persons?.find(p => p.state === 'fall')
  const live = !!selected && hasStream(selected, now)

  if (!selected) return <div className="card"><Empty icon="cam">Chưa có thiết bị nào. Cài dashboard_agent lên Pi để thêm camera.</Empty></div>

  const camNow = st?.camera ?? selected.camera ?? '0'
  async function doSwitch(i: number) {
    setBusy(true)
    const err = await switchCamera(selected!, i)
    setBusy(false)
    toast(err ? 'Không đổi được camera: ' + err : `Đã chuyển sang camera ${i}`, !!err)
  }

  return (
    <>
      <div className="cam-layout">
        <div className="card card-navy">
          <div className="card-head">
            <h2>{deviceName(selected)}</h2><DevicePill d={selected} now={now} />
            <div className="right">
              <div className="seg dark">
                {[0, 1].map(i => <button key={i} disabled={!live || busy} className={String(camNow) === String(i) ? 'on' : ''} onClick={() => doSwitch(i)}>Camera {i}</button>)}
              </div>
            </div>
          </div>
          <div className="cam-big">
            <StreamView d={selected} now={now} overlay={<>
              <span className="tag meta">{fmtTime(new Date(now).toISOString())}</span>
              {fallen && <span className="tag bottom">Người #{fallen.id} · nằm bất động {fmtDuration(Math.round(fallen.since))}</span>}
            </>} />
          </div>
          <div className="toolbar">
            <span className="muted">{selected.id}{st ? ` · ${fmtNumber(st.fps)} FPS` : ''} · cập nhật {fmtAgo(selected.last_seen, now)}</span>
            {live && <a className="btn btn-sm btn-ghost-white" href="#" onClick={async e => {
              e.preventDefault()
              const t = await accessToken()
              if (t) window.open(`${streamBase(selected)}/snapshot?token=${encodeURIComponent(t)}`, '_blank', 'noopener')
            }}><Icon name="img" />Chụp ảnh</a>}
            {live && <button className="btn btn-sm btn-ghost-white" onClick={() => {
              const img = document.querySelector<HTMLImageElement>('.cam-big img.stream')
              img?.requestFullscreen?.()
            }}><Icon name="expand" />Toàn màn hình</button>}
            <button className="btn btn-sm btn-danger" disabled={!live} onClick={async () => {
              const err = await resetAlarm(selected)
              toast(err ? 'Không tắt được báo động: ' + err : 'Đã gửi lệnh tắt báo động', !!err)
            }}><Icon name="mute" />Tắt báo động</button>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div className="card">
            <div className="card-head"><h2>Phân tích tư thế</h2><span className="sub">{st ? 'cập nhật mỗi giây' : 'chưa có dữ liệu trực tiếp'}</span></div>
            <div className="card-body"><ProbBars probs={st?.probs ?? null} /></div>
          </div>
          <div className="card">
            <div className="card-head"><h2>Thông tin thiết bị</h2></div>
            <div className="card-body">
              <dl className="kv">
                <dt>Mã thiết bị</dt><dd>{selected.id}</dd>
                <dt>Vị trí</dt><dd>{selected.location || '–'}</dd>
                <dt>Người trong khung</dt><dd className="num">{isOnline(selected, now) ? (st?.persons?.length ?? selected.persons_count) : '–'}</dd>
                <dt>Hộp ESP32</dt><dd><Esp32Pill d={selected} now={now} /></dd>
                <dt>Heartbeat</dt><dd>{fmtAgo(selected.last_seen, now)}</dd>
                <dt>Phần mềm agent</dt><dd>{selected.agent_version ? 'v' + selected.agent_version : '–'}</dd>
              </dl>
            </div>
          </div>
          <div className="card">
            <div className="card-head"><h2>Ca ngã đang mở</h2><span className="sub">{open.length}</span></div>
            {open.length === 0 ? <Empty icon="check">Không có ca nào chờ xử lý</Empty> : open.slice(0, 3).map(e => (
              <div key={e.id} className="card-body row" style={{ borderTop: '1px solid var(--line-2)' }}>
                <Thumb path={e.snapshot_path} />
                <div className="grow clickable" onClick={() => nav(`/alerts?id=${e.id}`)}>
                  <div style={{ fontWeight: 600, fontSize: 13.5 }}>Người #{e.person_id ?? '?'} · {fmtTime(e.fall_time || e.detected_at)}</div>
                  <div className="muted" style={{ fontSize: 12.5 }}>Chưa ai xác nhận</div>
                </div>
                <button className="btn btn-sm btn-primary" onClick={async () => {
                  const err = await updateEventStatus(e.id, 'acknowledged')
                  toast(err ? 'Lỗi: ' + err : 'Đã xác nhận ca ngã', !!err)
                }}>Xác nhận</button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {devices.length > 1 && (
        <div className="others">
          {devices.filter(d => d.id !== selected.id).map(d => (
            <div key={d.id} className="card mini" onClick={() => setParams({ device: d.id })}>
              <MiniPreview d={d} now={now} />
              <div className="mini-foot"><div><b>{deviceName(d)}</b><br /><span className="muted">{d.id}{isOnline(d, now) ? ` · ${d.persons_count} người` : ''}</span></div><DevicePill d={d} now={now} /></div>
            </div>
          ))}
        </div>
      )}
    </>
  )
}
