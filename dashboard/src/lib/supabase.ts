import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined
const key = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined

export const configError = !url || !key
  ? 'Thiếu VITE_SUPABASE_URL hoặc VITE_SUPABASE_ANON_KEY. Xem dashboard/.env.example.'
  : null

export const supabase = createClient(url || 'http://localhost', key || 'missing', {
  auth: { persistSession: true, autoRefreshToken: true },
  realtime: { params: { eventsPerSecond: 20 } },
})
