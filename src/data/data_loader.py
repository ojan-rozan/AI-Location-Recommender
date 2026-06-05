"""Load data dari Supabase (sebelumnya dari CSV lokal)."""

from pathlib import Path

from src.data import supabase_io as sio


def find_project_root() -> Path:
    """Cari root project (masih dipakai modul lain untuk path model/clean dll)."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "src").exists():
            return parent
    return Path.cwd()


ROOT = find_project_root()


class DataLoader:
    """Centralized data loader — sumber data: Supabase."""

    def load_cafes(self):
        """Load data cafe (hasil scraping Google Maps) dari Supabase."""
        df = sio.read_df(sio.RAW_CAFES)
        if df.empty:
            raise RuntimeError(
                f"Dataset '{sio.RAW_CAFES}' kosong di Supabase. "
            )
        return df

    def load_owner_stores(self):
        """Load data store owner dari Supabase."""
        return sio.read_df(sio.OWNER_STORES)

    def load_poi(self, category: str):
        """Load 1 kategori POI dari Supabase."""
        return sio.read_df(sio.raw_poi(category))

    def load_all_poi(self):
        """Load semua kategori POI sekaligus."""
        return {cat: self.load_poi(cat) for cat in sio.POI_CATEGORIES}

    def load_kecamatan_ref(self):
        """Load kecamatan reference dari Supabase."""
        return sio.read_df(sio.KECAMATAN_REF)

    def load_all(self) -> dict:
        """Load semua data."""
        print("Loading all data dari Supabase...")
        return {
            "cafes": self.load_cafes(),
            "owner": self.load_owner_stores(),
            "poi": self.load_all_poi(),
            "kecamatan_ref": self.load_kecamatan_ref(),
        }
