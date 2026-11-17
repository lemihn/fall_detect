import { createContext, useContext, useState, type ReactNode } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { Icon } from './Icon'
import { Brand } from './Brand'
import { useAuth, displayName, initials } from '../lib/auth'
import { useData } from '../lib/data'
import { deviceName, fmtDate, fmtDuration, fmtTime } from '../lib/format'
import { useToast } from './ui'

const NAV = [
  { to: '/', icon: 'home', label: 'Tổng quan', short: 'Tổng quan' },
  { to: '/camera', icon: 'cam', label: 'Camera trực tiếp', short: 'Camera' },
  { to: '/alerts', icon: 'bell', label: 'Cảnh báo', short: 'Cảnh báo' },
  { to: '/history', icon: 'hist', label: 'Lịch sử & thống kê', short: 'Lịch sử' },
  { to: '/devices', icon: 'dev', label: 'Thiết bị', short: 'Thiết bị' },
]

const TITLES: Record<string, [string, string]> = {
  '/': ['Tổng quan', ''],
  '/camera': ['Camera trực tiếp', 'Video qua kết nối bảo mật · chỉ người được cấp quyền mới xem được'],
  '/alerts': ['Cảnh báo', 'Danh sách ca té ngã và trạng thái xử lý'],
  '/history': ['Lịch sử & thống kê', 'Số liệu từ dữ liệu đã lưu trên hệ thống'],
  '/devices': ['Thiết bị', 'Các Raspberry Pi đang gửi dữ liệu lên hệ thống'],
  '/settings': ['Cài đặt', 'Thông báo, âm thanh và tài khoản'],
}

/* ---------- loc thiet bi dung chung (topbar) ---------- */
const FilterCtx = createContext<{ deviceId: string; setDeviceId: (id: string) => void }>({ deviceId: '', setDeviceId: () => {} })
export const useDeviceFilter = () => useContext(FilterCtx)

function AlertBar() {
  const { openEvents, deviceMap, updateEventStatus, now } = useData()
  const nav = useNavigate()
  const toast = useToast()
  const e = openEvents[0]
  if (!e) return null
  const d = deviceMap.get(e.device_id)
  const since = Math.max(0, (now - new Date(e.fall_time || e.detected_at).getTime()) / 1000)
  return (
    <div className="alert-bar" role="alert">
      <Icon name="warn" strokeWidth={2} />
      <div>
        <b>PHÁT HIỆN TÉ NGÃ · {deviceName(d, e.device_id)}</b>
        <div className="meta">
          Người #{e.person_id ?? '?'} · bắt đầu lúc {fmtTime(e.fall_time || e.detected_at)} · đã {fmtDuration(Math.round(since))}
          {openEvents.length > 1 && ` · còn ${openEvents.length - 1} ca khác chưa xử lý`}
        </div>
      </div>
      <div className="actions">
        <button className="btn btn-ghost-white" onClick={() => nav(`/camera?device=${encodeURIComponent(e.device_id)}`)}>Xem camera</button>
        <button className="btn btn-white" onClick={async () => {
          const err = await updateEventStatus(e.id, 'acknowledged')
          toast(err ? 'Không xác nhận được: ' + err : 'Đã xác nhận ca ngã', !!err)
        }}><Icon name="check" />Xác nhận đã biết</button>
      </div>
    </div>
  )
}

function Sidebar() {
  const { openEvents, realtime } = useData()
  return (
    <aside className="sidebar">
      <Brand />
      <nav className="nav">
        <div className="nav-label">Giám sát</div>
        {NAV.map(n => (
          <NavLink key={n.to} to={n.to} end={n.to === '/'} className={({ isActive }) => (isActive ? 'active' : '')}>
            <Icon name={n.icon} />{n.label}
            {n.to === '/alerts' && openEvents.length > 0 && <span className="count">{openEvents.length}</span>}
          </NavLink>
        ))}
        <div className="nav-label">Hệ thống</div>
        <NavLink to="/settings" className={({ isActive }) => (isActive ? 'active' : '')}><Icon name="gear" />Cài đặt</NavLink>
      </nav>
      <div className="sidebar-foot">
        <div><span className={`rt-dot ${realtime}`} />
          {realtime === 'live' ? 'Đồng bộ thời gian thực' : realtime === 'connecting' ? 'Đang kết nối…' : 'Mất kết nối, đang thử lại'}</div>
        <div className="copyright">© {new Date().getFullYear()} SMLab</div>
      </div>
    </aside>
  )
}

function Topbar({ title, sub }: { title: string; sub: ReactNode }) {
  const { devices, openEvents } = useData()
  const { profile, session } = useAuth()
  const { deviceId, setDeviceId } = useDeviceFilter()
  const nav = useNavigate()
  const name = displayName(profile, session?.user.email)
  return (
    <header className="topbar">
      <h1>{title}<small>{sub}</small></h1>
      <select className="select" value={deviceId} onChange={e => setDeviceId(e.target.value)} aria-label="Lọc thiết bị">
        <option value="">Tất cả thiết bị ({devices.length})</option>
        {devices.map(d => <option key={d.id} value={d.id}>{d.name || d.id}</option>)}
      </select>
      <span className="select" style={{ cursor: 'default', backgroundImage: 'none', paddingRight: 12 }}><Icon name="cal" />Hôm nay, {fmtDate(new Date().toISOString())}</span>
      <button className="icon-btn" onClick={() => nav('/alerts')} aria-label="Cảnh báo">
        <Icon name="bell" />{openEvents.length > 0 && <span className="badge">{openEvents.length}</span>}
      </button>
      <div className="user"><div className="avatar">{initials(name)}</div><div><b>{name}</b><span>{profile?.role === 'admin' ? 'Quản trị viên' : 'Người theo dõi'}</span></div></div>
    </header>
  )
}

function MobileTop({ title }: { title: string }) {
  const { openEvents } = useData()
  const nav = useNavigate()
  return (
    <div className="mtop">
      <div className="brand-mark"><Icon name="logo" strokeWidth={2.2} /></div>
      <div className="mtop-title"><small>SMLab</small><b>{title}</b></div>
      <button className="icon-btn" onClick={() => nav('/settings')} aria-label="Cài đặt" style={{ marginLeft: 'auto' }}><Icon name="gear" /></button>
      <button className="icon-btn" onClick={() => nav('/alerts')} aria-label="Cảnh báo" style={{ marginLeft: 0 }}>
        <Icon name="bell" />{openEvents.length > 0 && <span className="badge">{openEvents.length}</span>}
      </button>
    </div>
  )
}

function TabBar() {
  const { openEvents } = useData()
  return (
    <nav className="tabbar">
      {NAV.map(n => (
        <NavLink key={n.to} to={n.to} end={n.to === '/'} className={({ isActive }) => (isActive ? 'active' : '')}>
          <Icon name={n.icon} />
          {n.to === '/alerts' && openEvents.length > 0 && <span className="count">{openEvents.length}</span>}
          {n.short}
        </NavLink>
      ))}
    </nav>
  )
}

export function Layout() {
  const loc = useLocation()
  const { loading, error, now } = useData()
  const [deviceId, setDeviceId] = useState('')
  const [title, sub] = TITLES[loc.pathname] || ['', '']
  const subText = loc.pathname === '/' ? `Cập nhật lúc ${fmtTime(new Date(now).toISOString())} · tự động làm mới` : sub
  return (
    <FilterCtx.Provider value={{ deviceId, setDeviceId }}>
      <div className="app">
        <Sidebar />
        <div className="main">
          <MobileTop title={title} />
          <AlertBar />
          <Topbar title={title} sub={subText} />
          <div className="content">
            {error && <div className="err-box">Không tải được dữ liệu: {error}</div>}
            {loading ? <div className="empty"><div className="spinner" />Đang tải dữ liệu…</div> : <Outlet />}
          </div>
        </div>
        <TabBar />
      </div>
    </FilterCtx.Provider>
  )
}
