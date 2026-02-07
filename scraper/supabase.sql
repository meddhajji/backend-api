CREATE TABLE IF NOT EXISTS laptops (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    price NUMERIC DEFAULT 0,
    city TEXT,
    link TEXT UNIQUE NOT NULL,
    brand TEXT,
    model TEXT,
    model_num TEXT,
    cpu TEXT,
    ram TEXT,
    storage TEXT,
    gpu TEXT,
    gpu_vram TEXT,
    screen_size NUMERIC,
    refresh_rate INTEGER,
    is_new BOOLEAN DEFAULT FALSE,
    is_touchscreen BOOLEAN DEFAULT FALSE,
    is_shop BOOLEAN DEFAULT FALSE,
    has_delivery BOOLEAN DEFAULT FALSE,
    is_ssd BOOLEAN DEFAULT FALSE,
    cpu_family TEXT,
    cpu_generation INTEGER,
    gpu_type TEXT,
    gpu_family TEXT,
    ram_gb INTEGER DEFAULT 0,
    storage_gb INTEGER DEFAULT 0,
    cpu_score INTEGER DEFAULT 0,
    gpu_score INTEGER DEFAULT 0,
    ram_score INTEGER DEFAULT 0,
    storage_score INTEGER DEFAULT 0,
    screen_score INTEGER DEFAULT 0,
    condition_score INTEGER DEFAULT 0,
    laptop_score INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_laptops_link ON laptops(link);
CREATE INDEX IF NOT EXISTS idx_laptops_brand ON laptops(brand);
CREATE INDEX IF NOT EXISTS idx_laptops_price ON laptops(price);
CREATE INDEX IF NOT EXISTS idx_laptops_score ON laptops(laptop_score);