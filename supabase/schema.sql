-- =============================================================================
-- Fall Detection Dashboard - Supabase schema
-- -----------------------------------------------------------------------------
-- Chay 1 lan tren project Supabase (SQL Editor hoac MCP apply_migration).
-- Co the chay lai an toan: moi lenh deu kiem tra "if not exists" / "or replace".
--
-- Luong du lieu:
--   dashboard_agent (tren Pi, dung service_role key)  --ghi-->  devices, events, storage
--   website (anon key + dang nhap)                    --doc/xu ly-->  devices, events
--
-- service_role bo qua RLS nen agent ghi duoc tat ca.
-- Nguoi dung website chi thay du lieu khi co ten trong bang dashboard_users.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Nguoi duoc phep dung dashboard
--    Tao tai khoan trong Authentication > Users, roi them vao bang nay
--    (xem supabase/README.md). Tai khoan tu dang ky ma khong co trong bang
--    nay se KHONG thay du lieu nao.
-- -----------------------------------------------------------------------------
create table if not exists public.dashboard_users (
  user_id    uuid primary key references auth.users (id) on delete cascade,
  email      text not null,
  full_name  text not null default '',
  role       text not null default 'viewer' check (role in ('admin', 'viewer')),
  created_at timestamptz not null default now()
);

comment on table public.dashboard_users is
  'Danh sach tai khoan duoc xem dashboard. admin: sua thong tin thiet bi; viewer: xem + xu ly canh bao.';


-- Ham kiem tra quyen, dung trong cac policy.
-- security definer de doc duoc dashboard_users ma khong bi RLS chan vong lap.
-- Dat trong schema 'private' (khong mo ra API) de khong goi duoc qua /rest/v1/rpc.
create schema if not exists private;
revoke all on schema private from public, anon;
grant usage on schema private to authenticated;

create or replace function private.is_dashboard_user()
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1 from public.dashboard_users where user_id = (select auth.uid())
  );
$$;

create or replace function private.is_dashboard_admin()
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1 from public.dashboard_users
    where user_id = (select auth.uid()) and role = 'admin'
  );
$$;

revoke all on function private.is_dashboard_user()  from public, anon;
revoke all on function private.is_dashboard_admin() from public, anon;
grant execute on function private.is_dashboard_user()  to authenticated;
grant execute on function private.is_dashboard_admin() to authenticated;


-- -----------------------------------------------------------------------------
-- 2. Thiet bi (moi Pi/camera la 1 dong)
--    Agent cap nhat (upsert) dong nay moi ~10 giay = heartbeat.
--    Website coi thiet bi la "offline" neu last_seen qua cu (> 30 giay).
-- -----------------------------------------------------------------------------
create table if not exists public.devices (
  id               text primary key,                    -- do agent dat, vd: 'pi-phong-khach'
  name             text not null default '',            -- ten hien thi (admin sua tren web)
  location         text not null default '',            -- vi tri (admin sua tren web)

  -- Trang thai lay tu /status cua 4_realtime.py
  state            text not null default 'unknown'
                   check (state in ('normal', 'sit', 'lie', 'fall', 'warmup', 'unknown')),
  alarm            smallint not null default 0 check (alarm between 0 and 2),  -- 0 binh thuong, 1 canh bao, 2 khan cap
  fps              real,
  camera           text,                                -- nguon camera dang dung
  persons_count    smallint not null default 0,
  last_seen        timestamptz,                         -- lan heartbeat gan nhat

  -- Camera live qua gateway + Cloudflare Tunnel
  stream_url       text,
  stream_updated_at timestamptz,

  -- Hop bao dong ESP32 (agent doc /data cua ESP32 roi gui kem)
  esp32_online     boolean not null default false,
  esp32_state      text,                                -- normal / warning / fall
  esp32_last_seen  timestamptz,
  esp32_battery    smallint check (esp32_battery between 0 and 100),  -- de danh khi firmware co do pin

  agent_version    text,
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now()
);

comment on table public.devices is 'Moi dong la 1 thiet bi Pi. Agent upsert dinh ky (heartbeat).';


-- -----------------------------------------------------------------------------
-- 3. Su kien te nga
--    Agent tao 1 dong khi 1 nguoi nam bat dong du lau (cung luc voi Telegram).
--    id do agent tu sinh (uuid) de gui lai khi mat mang khong bi trung.
-- -----------------------------------------------------------------------------
create table if not exists public.events (
  id               uuid primary key default gen_random_uuid(),
  device_id        text not null references public.devices (id) on delete cascade,
  person_id        integer,                             -- ID nguoi theo tracker
  type             text not null default 'fall' check (type in ('fall')),
  fall_time        timestamptz,                         -- thoi diem bat dau nga
  detected_at      timestamptz not null default now(),  -- thoi diem phat canh bao
  duration_s       real,                                -- so giay nam bat dong luc canh bao
  snapshot_path    text,                                -- duong dan anh trong bucket 'snapshots'

  status           text not null default 'new'
                   check (status in ('new', 'acknowledged', 'resolved')),
  acknowledged_at  timestamptz,
  acknowledged_by  uuid references auth.users (id) on delete set null,
  resolved_at      timestamptz,
  resolved_by      uuid references auth.users (id) on delete set null,
  note             text not null default '',

  created_at       timestamptz not null default now()
);

comment on table public.events is 'Moi dong la 1 ca te nga. Nguoi dung website chuyen trang thai new -> acknowledged -> resolved.';

create index if not exists events_detected_at_idx on public.events (detected_at desc);
create index if not exists events_device_detected_idx on public.events (device_id, detected_at desc);
create index if not exists events_open_idx on public.events (detected_at desc) where status <> 'resolved';
create index if not exists events_acknowledged_by_idx on public.events (acknowledged_by);
create index if not exists events_resolved_by_idx on public.events (resolved_by);


-- -----------------------------------------------------------------------------
-- 4. Trigger
-- -----------------------------------------------------------------------------

-- devices.updated_at tu dong
create or replace function public.set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists devices_set_updated_at on public.devices;
create trigger devices_set_updated_at
  before update on public.devices
  for each row execute function public.set_updated_at();


-- events: khi doi trang thai, tu ghi ai xu ly va luc nao (nguoi dung khong tu dien duoc)
create or replace function public.events_track_status()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  if new.status is distinct from old.status then
    if new.status = 'new' then
      new.acknowledged_at := null;  new.acknowledged_by := null;
      new.resolved_at := null;      new.resolved_by := null;
    end if;
    if new.status in ('acknowledged', 'resolved') and old.acknowledged_at is null then
      new.acknowledged_at := now();
      new.acknowledged_by := auth.uid();
    end if;
    if new.status = 'resolved' then
      new.resolved_at := now();
      new.resolved_by := auth.uid();
    elsif old.status = 'resolved' then
      new.resolved_at := null;      new.resolved_by := null;
    end if;
  end if;
  return new;
end;
$$;

drop trigger if exists events_track_status on public.events;
create trigger events_track_status
  before update on public.events
  for each row execute function public.events_track_status();


-- -----------------------------------------------------------------------------
-- 5. Quyen truy cap (RLS + gioi han cot duoc sua)
-- -----------------------------------------------------------------------------
alter table public.dashboard_users enable row level security;
alter table public.devices         enable row level security;
alter table public.events          enable row level security;

-- anon (chua dang nhap): khong co quyen gi
revoke all on public.dashboard_users, public.devices, public.events from anon;

-- authenticated: chi doc + sua mot so cot nhat dinh
revoke all on public.dashboard_users, public.devices, public.events from authenticated;
grant select on public.dashboard_users, public.devices, public.events to authenticated;
grant update (name, location)     on public.devices to authenticated;
grant update (status, note)       on public.events  to authenticated;

-- dashboard_users: moi nguoi xem duoc chinh minh; admin xem duoc tat ca
drop policy if exists dashboard_users_select on public.dashboard_users;
create policy dashboard_users_select on public.dashboard_users
  for select to authenticated
  using (user_id = (select auth.uid()) or (select private.is_dashboard_admin()));

-- devices
drop policy if exists devices_select on public.devices;
create policy devices_select on public.devices
  for select to authenticated
  using ((select private.is_dashboard_user()));

drop policy if exists devices_update on public.devices;
create policy devices_update on public.devices
  for update to authenticated
  using ((select private.is_dashboard_admin()))
  with check ((select private.is_dashboard_admin()));

-- events
drop policy if exists events_select on public.events;
create policy events_select on public.events
  for select to authenticated
  using ((select private.is_dashboard_user()));

drop policy if exists events_update on public.events;
create policy events_update on public.events
  for update to authenticated
  using ((select private.is_dashboard_user()))
  with check ((select private.is_dashboard_user()));


-- -----------------------------------------------------------------------------
-- 6. Storage: anh hien truong (private)
--    Agent upload bang service_role. Website xem qua signed URL.
-- -----------------------------------------------------------------------------
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('snapshots', 'snapshots', false, 1048576, array['image/jpeg'])
on conflict (id) do update
  set public = excluded.public,
      file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;

drop policy if exists snapshots_select on storage.objects;
create policy snapshots_select on storage.objects
  for select to authenticated
  using (bucket_id = 'snapshots' and (select private.is_dashboard_user()));


-- -----------------------------------------------------------------------------
-- 7. Realtime: website nhan thay doi ngay lap tuc (van tuan theo RLS o tren)
-- -----------------------------------------------------------------------------
do $$
begin
  if not exists (select 1 from pg_publication_tables
                 where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = 'devices') then
    alter publication supabase_realtime add table public.devices;
  end if;
  if not exists (select 1 from pg_publication_tables
                 where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = 'events') then
    alter publication supabase_realtime add table public.events;
  end if;
end;
$$;
