import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { supabase } from './supabase'
import type { Device, EventStatus, FallEvent } from './types'
import { deviceName } from './format'
import { notifyNewFall } from './notify'

/** Tai su kien trong N ngay gan nhat; du cho trang lich su 30 ngay */
const HISTORY_DAYS = 90
const MAX_EVENTS = 2000

interface DataState {
  loading: boolean
  error: string | null
  realtime: 'connecting' | 'live' | 'offline'
  devices: Device[]
  deviceMap: Map<string, Device>
  events: FallEvent[]
  openEvents: FallEvent[]
  /** dong ho chung, cap nhat moi 5 giay de tinh online/offline, "x giay truoc" */
  now: number
  updateEventStatus: (id: string, status: EventStatus) => Promise<string | null>
  updateEventNote: (id: string, note: string) => Promise<string | null>
  updateDevice: (id: string, patch: { name?: string; location?: string }) => Promise<string | null>
}

const Ctx = createContext<DataState | null>(null)

function upsertBy<T extends { id: string }>(list: T[], row: T): T[] {
  const i = list.findIndex(x => x.id === row.id)
  if (i < 0) return [row, ...list]
  const copy = list.slice()
  copy[i] = { ...copy[i], ...row }
  return copy
}

export function DataProvider({ children }: { children: ReactNode }) {
  const [devices, setDevices] = useState<Device[]>([])
  const [events, setEvents] = useState<FallEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [realtime, setRealtime] = useState<DataState['realtime']>('connecting')
  const [now, setNow] = useState(Date.now())
  const devicesRef = useRef(devices)
  devicesRef.current = devices

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 5000)
    return () => clearInterval(t)
  }, [])

  const loadAll = useCallback(async () => {
    const since = new Date(Date.now() - HISTORY_DAYS * 86400_000).toISOString()
    const [d, e] = await Promise.all([
      supabase.from('devices').select('*').order('name'),
      supabase.from('events').select('*').gte('detected_at', since).order('detected_at', { ascending: false }).limit(MAX_EVENTS),
    ])
    if (d.error || e.error) {
      setError((d.error || e.error)!.message)
    } else {
      setError(null)
      setDevices(d.data as Device[])
      setEvents(e.data as FallEvent[])
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    loadAll()
    const ch = supabase
      .channel('dashboard')
      .on('postgres_changes', { event: '*', schema: 'public', table: 'devices' }, payload => {
        if (payload.eventType === 'DELETE') {
          const id = (payload.old as { id?: string }).id
          setDevices(list => list.filter(x => x.id !== id))
        } else {
          setDevices(list => upsertBy(list, payload.new as Device))
        }
      })
      .on('postgres_changes', { event: '*', schema: 'public', table: 'events' }, payload => {
        if (payload.eventType === 'DELETE') {
          const id = (payload.old as { id?: string }).id
          setEvents(list => list.filter(x => x.id !== id))
          return
        }
        const row = payload.new as FallEvent
        setEvents(list => upsertBy(list, row).sort((a, b) => b.detected_at.localeCompare(a.detected_at)))
        if (payload.eventType === 'INSERT') {
          const dev = devicesRef.current.find(x => x.id === row.device_id)
          notifyNewFall(deviceName(dev, row.device_id), row)
        }
      })
      .subscribe(status => {
        if (status === 'SUBSCRIBED') {
          setRealtime('live')
          loadAll() // bu cac thay doi trong luc mat ket noi
        } else if (status === 'CHANNEL_ERROR' || status === 'TIMED_OUT' || status === 'CLOSED') {
          setRealtime('offline')
        }
      })
    return () => { supabase.removeChannel(ch) }
  }, [loadAll])

  const value = useMemo<DataState>(() => {
    const deviceMap = new Map(devices.map(d => [d.id, d]))
    return {
      loading, error, realtime, devices, deviceMap, events, now,
      openEvents: events.filter(e => e.status === 'new'),
      async updateEventStatus(id, status) {
        const { data, error } = await supabase.from('events').update({ status }).eq('id', id).select().maybeSingle()
        if (error) return error.message
        if (data) setEvents(list => upsertBy(list, data as FallEvent))
        return null
      },
      async updateEventNote(id, note) {
        const { data, error } = await supabase.from('events').update({ note }).eq('id', id).select().maybeSingle()
        if (error) return error.message
        if (data) setEvents(list => upsertBy(list, data as FallEvent))
        return null
      },
      async updateDevice(id, patch) {
        const { data, error } = await supabase.from('devices').update(patch).eq('id', id).select().maybeSingle()
        if (error) return error.message
        if (!data) return 'Không có quyền sửa (chỉ admin).'
        setDevices(list => upsertBy(list, data as Device))
        return null
      },
    }
  }, [loading, error, realtime, devices, events, now])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useData(): DataState {
  const v = useContext(Ctx)
  if (!v) throw new Error('useData ngoai DataProvider')
  return v
}
