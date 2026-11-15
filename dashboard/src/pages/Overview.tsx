import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Icon } from '../components/Icon'
import { BarChart } from '../components/charts'
import { DevicePill, Empty, Esp32Pill, StreamView, useToast } from '../components/ui'
import { useDeviceFilter } from '../components/Layout'
import { useData } from '../lib/data'
import { resetAlarm } from '../lib/gateway'
import { addDays, avgResponse, countsPerDay, inRange, startOfDay } from '../lib/stats'
import { deviceName, deviceView, fmtAgo, fmtDuration, fmtNumber, fmtTime, isOnline } from '../lib/format'
import type { Device, FallEvent } from '../lib/types'

type FeedItem = { at: string; kind: 'fall' | 'done' | 'ack' | 'off'; title: string; desc: string; event?: FallEvent }

export function OverviewPage() {
  const { devices, events, now, deviceMap, updateEventStatus } = useData()
  const { deviceId } = useDeviceFilter()
  const nav = useNavigate()
  const toast = useToast()
  const [range, setRange] = useState<7 | 30>(7)

  const devs = deviceId ? devices.filter(d => d.id === deviceId) : devices
  const evs = deviceId ? events.filter(e => e.device_id === deviceId) : events

  const k = useMemo(() => {
    const today = startOfDay(new Date(now))
    const tomorrow = addDays(today, 1), yesterday = addDays(today, -1)
    const todayN = evs.filter(e => inRange(e, today, tomorrow)).length
    const yN = evs.filter(e => inRange(e, yesterday, today)).length
    const open = evs.filter(e => e.status === 'new')
    const oldest = open.length ? open[open.length - 1] : null
    const online = devs.filter(d => isOnline(d, now))
    const offline = devs.filter(d => !isOnline(d, now))
    const w1 = avgResponse(evs.filter(e => inRange(e, addDays(today, -6), tomorrow)))
    const w0 = avgResponse(evs.filter(e => inRange(e, addDays(today, -13), addDays(today, -6))))
    return { todayN, yN, open, oldest, online, offline, w1, w0 }
  }, [evs, devs, now])

  // Camera noi bat: thiet bi co ca nga dang mo > dang te nga > online dau tien
  const featured: Device | undefined = useMemo(() => {
    const byOpen = k.open[0] && deviceMap.get(k.open[0].device_id)
    return byOpen || devs.find(d => deviceView(d, now) === 'fall') || devs.find(d => isOnline(d, now)) || devs[0]
  }, [k.open, deviceMap, devs, now])

  const feed: FeedItem[] = useMemo(() => {
    const items: FeedItem[] = evs.slice(0, 8).map(e => {
      const name = deviceName(deviceMap.get(e.device_id), e.device_id)
      if (e.status === 'resolved') return { at: e.resolved_at || e.detected_at, kind: 'done', title: `Đã xử lý · ${name}`, desc: e.note ? `"${e.note}"` : `Ca ngã lúc ${fmtTime(e.fall_time || e.detected_at, false)}`, event: e }
      if (e.status === 'acknowledged') return { at: e.acknowledged_at || e.detected_at, kind: 'ack', title: `Đã xác nhận · ${name}`, desc: `Người #${e.person_id ?? '?'} · đang xử lý`, event: e }
      return { at: e.detected_at, kind: 'fall', title: `Té ngã · ${name}`, desc: `Người #${e.person_id ?? '?'} bất động ${fmtDuration(e.duration_s)} khi cảnh báo`, event: e }
    })
    for (const d of k.offline) if (d.last_seen) items.push({ at: d.last_seen, kind: 'off', title: `Mất kết nối · ${deviceName(d)}`, desc: `Không nhận heartbeat từ ${fmtTime(d.last_seen, false)}` })
    return items.sort((a, b) => b.at.localeCompare(a.at)).slice(0, 6)
  }, [evs, k.offline, deviceMap])

  const chart = useMemo(() => countsPerDay(evs, range, new Date(now)), [evs, range, now])
  const total = chart.reduce((a, b) => a + b.count, 0)
  const respDelta = k.w1 != null && k.w0 != null ? Math.round(k.w1 - k.w0) : null

  return (
    <>
      <div className="kpis">
        <div className={`card kpi${k.todayN ? ' danger' : ''}`}>
          <div className="label"><Icon name="warn" />Ca ngã hôm nay</div>
          <div className="value num">{k.todayN}</div>
          <div className="foot">{k.todayN - k.yN !== 0 && <span className={`delta ${k.todayN > k.yN ? 'up-bad' : 'down-good'}`}>{k.todayN > k.yN ? '+' : ''}{k.todayN - k.yN}</span>}so với hôm qua ({k.yN})</div>
        </div>
        <div className="card kpi">
          <div className="label"><Icon name="bell" />Chưa xử lý</div>
          <div className="value num">{k.open.length}</div>
          <div className="foot">{k.oldest ? `cũ nhất ${fmtAgo(k.oldest.detected_at, now)}` : 'không có ca nào đang chờ'}</div>
        </div>
        <div className="card kpi">
          <div className="label"><Icon name="wifi" />Thiết bị online</div>
          <div className="value num">{k.online.length}<small> / {devs.length}</small></div>
          <div className="foot">{k.offline.length ? <><span className="delta neutral">{k.offline.length} offline</span><span className="ellipsis">{k.offline.map(d => deviceName(d)).join(', ')}</span></> : 'tất cả đang hoạt động'}</div>
        </div>
        <div className="card kpi">
          <div className="label"><Icon name="clock" />Phản hồi trung bình</div>
          <div className="value num">{k.w1 != null ? fmtDuration(Math.round(k.w1)) : '–'}</div>
          <div className="foot">{respDelta != null && respDelta !== 0 && <span className={`delta ${respDelta < 0 ? 'down-good' : 'up-bad'}`}>{respDelta > 0 ? '+' : '−'}{fmtDuration(Math.abs(respDelta))}</span>}trong 7 ngày</div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card card-navy">
          <div className="card-head">
            <h2>{featured ? deviceName(featured) : 'Camera'}</h2>
            {featured && <DevicePill d={featured} now={now} />}
            <div className="right"><button className="btn btn-sm btn-ghost-white" onClick={() => nav(`/camera${featured ? '?device=' + encodeURIComponent(featured.id) : ''}`)}><Icon name="expand" />Mở rộng</button></div>
          </div>
          <div className="cam-wrap"><StreamView d={featured} now={now} overlay={<span className="tag meta">{fmtTime(new Date(now).toISOString())}</span>} /></div>
          <div className="cam-foot">
            <span className="muted">{!featured ? 'Chưa có thiết bị' : isOnline(featured, now)
              ? `${featured.id}${featured.fps ? ` · ${fmtNumber(featured.fps)} FPS` : ''} · ${featured.persons_count} người trong khung hình`
              : `${featured.id} · mất kết nối ${fmtAgo(featured.last_seen, now)}`}</span>
            {featured && isOnline(featured, now) && featured.stream_url && (
              <button className="btn btn-sm btn-danger" onClick={async () => {
                const err = await resetAlarm(featured)
                toast(err ? 'Không tắt được báo động: ' + err : 'Đã gửi lệnh tắt báo động', !!err)
              }}><Icon name="mute" />Tắt báo động</button>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-head"><h2>Hoạt động gần đây</h2><div className="right"><Link className="link" to="/alerts">Xem tất cả</Link></div></div>
          {feed.length === 0 ? <Empty>Chưa có hoạt động nào</Empty> : (
            <ul className="feed">
              {feed.map((f, i) => (
                <li key={i}>
                  <span className={`ic ${f.kind === 'fall' ? 'red' : f.kind === 'ack' ? 'amber' : f.kind === 'done' ? 'green' : 'gray'}`}>
                    <Icon name={f.kind === 'fall' ? 'warn' : f.kind === 'done' || f.kind === 'ack' ? 'check' : 'wifioff'} />
                  </span>
                  <div className="grow">
                    <div className="t">{f.title}</div>
                    <div className="d ellipsis">{f.desc}</div>
                    {f.kind === 'fall' && f.event && f.event.status === 'new' && (
                      <div className="acts">
                        <button className="btn btn-sm btn-primary" onClick={async () => {
                          const err = await updateEventStatus(f.event!.id, 'acknowledged')
                          toast(err ? 'Lỗi: ' + err : 'Đã xác nhận ca ngã', !!err)
                        }}>Xác nhận</button>
                        <button className="btn btn-sm btn-outline" onClick={() => nav(`/alerts?id=${f.event!.id}`)}>Xem chi tiết</button>
                      </div>
                    )}
                  </div>
                  <span className="when">{fmtTime(f.at, false)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-head"><h2>Thiết bị</h2><span className="sub">{devs.length} thiết bị</span><div className="right"><Link className="btn btn-sm btn-outline" to="/devices">Quản lý</Link></div></div>
          {devs.length === 0 ? <Empty icon="dev">Chưa có thiết bị nào gửi dữ liệu. Cài dashboard_agent lên Pi để thiết bị tự xuất hiện.</Empty> : (
            <div style={{ overflowX: 'auto' }}>
              <table>
                <thead><tr><th>Thiết bị</th><th>Trạng thái</th><th>Người</th><th className="hide-sm">FPS</th><th className="hide-sm">Hộp ESP32</th><th>Cập nhật</th></tr></thead>
                <tbody>
                  {devs.map(d => (
                    <tr key={d.id} className="clickable" onClick={() => nav(`/camera?device=${encodeURIComponent(d.id)}`)}>
                      <td><b style={{ display: 'block', fontWeight: 600 }}>{deviceName(d)}</b><span className="muted" style={{ fontSize: 12 }}>{d.id}{d.location ? ` · ${d.location}` : ''}</span></td>
                      <td><DevicePill d={d} now={now} /></td>
                      <td className="num">{isOnline(d, now) ? d.persons_count : <span className="faint">–</span>}</td>
                      <td className="num hide-sm">{isOnline(d, now) && d.fps ? fmtNumber(d.fps) : <span className="faint">–</span>}</td>
                      <td className="hide-sm"><Esp32Pill d={d} now={now} /></td>
                      <td className="muted">{fmtAgo(d.last_seen, now)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-head"><h2>Ca ngã {range} ngày qua</h2>
            <div className="right"><div className="seg"><button className={range === 7 ? 'on' : ''} onClick={() => setRange(7)}>7 ngày</button><button className={range === 30 ? 'on' : ''} onClick={() => setRange(30)}>30 ngày</button></div></div>
          </div>
          <div className="card-body">
            <BarChart data={chart} height={190} labelEvery={range === 7 ? 1 : 5} />
            <div className="legend" style={{ marginTop: 6 }}><span><i style={{ background: '#1f4675' }} />Ca ngã</span><span><i style={{ background: '#dc2626' }} />Hôm nay</span><span style={{ marginLeft: 'auto' }}>Tổng {range} ngày: <b style={{ color: 'var(--text)' }}>{total}</b></span></div>
          </div>
        </div>
      </div>
    </>
  )
}
