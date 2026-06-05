do $$
declare
    t text;
    tables text[] := array[
        'raw_cafes',
        'owner_stores',
        'kecamatan_ref',
        'raw_poi_office',
        'raw_poi_mall',
        'raw_poi_transit',
        'raw_poi_school',
        'clean_cafes',
        'clean_owner',
        'clean_poi_office',
        'clean_poi_mall',
        'clean_poi_transit',
        'clean_poi_school'
    ];
begin
    foreach t in array tables
    loop
        execute format(
            'create table if not exists %I (
                id          bigint generated always as identity primary key,
                data        jsonb       not null,
                created_at  timestamptz default now()
            );', t
        );
    end loop;
end $$;
