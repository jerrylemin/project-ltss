from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import urllib.request


SNAP_GRAPHS = {
    "roadNet-CA": "https://snap.stanford.edu/data/roadNet-CA.txt.gz",
    "com-youtube": "https://snap.stanford.edu/data/com-youtube.ungraph.txt.gz",
    "wiki-talk": "https://snap.stanford.edu/data/wiki-Talk.txt.gz",
    "amazon0601": "https://snap.stanford.edu/data/amazon0601.txt.gz",
    "soc-livejournal": "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
}

SNAP_GRAPH_FALLBACKS = {
    "com-youtube": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
}


def _write_graph_tsv(name: str, url: str, output_dir: Path, *, force: bool) -> tuple[int, int]:
    output_path = output_dir / f"{name}.tsv"
    temp_path = output_dir / f"{name}.tsv.tmp"
    if output_path.exists() and not force:
        node_count, edge_count = _count_existing_graph(output_path)
        _validate_graph(name, output_path, edge_count)
        return node_count, edge_count

    urls = [url]
    fallback_url = SNAP_GRAPH_FALLBACKS.get(name)
    if fallback_url:
        urls.append(fallback_url)

    last_error: Exception | None = None
    for candidate_url in urls:
        nodes: set[int] = set()
        edge_count = 0
        try:
            if temp_path.exists():
                temp_path.unlink()
            with urllib.request.urlopen(candidate_url) as response:
                with gzip.GzipFile(fileobj=response) as gzip_stream:
                    with temp_path.open("w", encoding="utf-8", newline="\n") as writer:
                        for raw_line in gzip_stream:
                            line = raw_line.decode("utf-8").strip()
                            if not line or line.startswith("#"):
                                continue
                            parts = line.replace(",", " ").split()
                            if len(parts) < 2:
                                continue
                            src = int(parts[0])
                            dst = int(parts[1])
                            writer.write(f"{src}\t{dst}\n")
                            nodes.add(src)
                            nodes.add(dst)
                            edge_count += 1
                            if edge_count % 1_000_000 == 0:
                                print(f"{name}: {edge_count:,} edges written...")
            _validate_graph(name, temp_path, edge_count)
            temp_path.replace(output_path)
            return len(nodes), edge_count
        except Exception as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            if candidate_url != urls[-1]:
                print(f"{name}: primary URL failed ({exc}); trying fallback {urls[-1]}")
    assert last_error is not None
    raise last_error


def _count_existing_graph(path: Path) -> tuple[int, int]:
    nodes: set[int] = set()
    edge_count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            src = int(parts[0])
            dst = int(parts[1])
            nodes.add(src)
            nodes.add(dst)
            edge_count += 1
    return len(nodes), edge_count


def _validate_graph(name: str, path: Path, edge_count: int) -> None:
    if edge_count <= 0:
        raise ValueError(f"{name}: downloaded graph has no edge rows: {path}")
    if not path.exists() or path.stat().st_size <= 0:
        raise ValueError(f"{name}: downloaded graph file is empty: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the five required SNAP PageRank graphs.")
    parser.add_argument("--output-dir", default="data/graphs", help="Directory for decompressed TSV files.")
    parser.add_argument("--force", action="store_true", help="Re-download graphs even when TSV files already exist.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, url in SNAP_GRAPHS.items():
        output_path = output_dir / f"{name}.tsv"
        action = "downloading" if args.force or not output_path.exists() else "using existing"
        print(f"{name}: {action} {url}")
        node_count, edge_count = _write_graph_tsv(name, url, output_dir, force=args.force)
        print(f"{name}: nodes={node_count} edges={edge_count} path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
