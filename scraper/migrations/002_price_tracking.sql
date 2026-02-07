-- Phase 9: Schema updates for price tracking and status
-- Run these ALTER TABLE commands in Supabase SQL Editor

-- Add last_price column for price change tracking
ALTER TABLE laptops 
ADD COLUMN IF NOT EXISTS last_price NUMERIC DEFAULT NULL;

-- Add is_sold flag for items no longer available
ALTER TABLE laptops 
ADD COLUMN IF NOT EXISTS is_sold BOOLEAN DEFAULT FALSE;

-- Add is_new_listing flag for newly scraped items
ALTER TABLE laptops 
ADD COLUMN IF NOT EXISTS is_new_listing BOOLEAN DEFAULT FALSE;

-- Add index for sold status filtering
CREATE INDEX IF NOT EXISTS idx_laptops_is_sold ON laptops(is_sold);

-- Add index for new listing filtering  
CREATE INDEX IF NOT EXISTS idx_laptops_is_new_listing ON laptops(is_new_listing);

-- Comment: Price tracking logic:
-- 1. Before scrape: price = current, last_price = null (or previous change)
-- 2. After scrape: if price changed, last_price = old price, price = new price
-- 3. UI shows: current price, and if last_price exists: "was X DH" crossed out
