"""The workspace's folders as Settings lists them, each cleared as a whole.

Clearing a folder deletes what is in it the way deleting each entry would:
a set with its history, a dataset with the history of its questions, a
batch, a model the Studio made (``ModelRemoval``), an ended job's entry.
What is in use is refused as deleting it alone would be (409 or busy), and
then nothing is cleared, except for jobs: running ones stay and the rest
go. Nothing outside the workspace is touched.
"""

import asyncio
from pathlib import Path
from typing import Any

from ..picker import folder_size
from .batch_results import Results
from .batches import Batches
from .jobs import Jobs
from .removal import ModelRemoval
from .workspace import Conflict, Workspace

# The folders listed, in this order, with what each holds.
FOLDERS = {
    "sets": "Question sets, with their history",
    "history": "Every run in the Playground, and its images",
    "data": "Datasets, with the history of their questions",
    "batches": "Batches and their answers",
    "models": "Models the Studio converted or fitted",
    "jobs": "Fits, evaluations, downloads and conversions as Training lists them",
    "speed": "How long answers took on this Mac (the Models page's Speed)",
}


class Clearing:
    def __init__(
        self,
        files: Workspace,
        jobs: Jobs,
        batches: Batches,
        removal: ModelRemoval,
        results: Results,
    ):
        self.files = files
        self.jobs = jobs
        self.batches = batches
        self.removal = removal
        self.results = results

    def folders(self) -> list[dict[str, Any]]:
        """Each folder: ``folder``, ``what``, ``path``, ``bytes``, ``count`` (its
        entries as the Studio lists them) (blocks)."""
        counts = {
            "sets": len(self.files.list_sets()),
            "history": sum(len(paths) for paths in self._histories()),
            "data": len(self.files.list_data()),
            "batches": len(self.batches.list()),
            "models": len(_visible(self.files.root / "models")),
            "jobs": len(_visible(self.files.root / "jobs")),
            "speed": len(_visible(self.files.root / "speed")),
        }
        return [
            {
                "folder": folder,
                "what": what,
                "path": str(self.files.root / folder),
                "bytes": folder_size(self.files.root / folder),
                "count": counts[folder],
            }
            for folder, what in FOLDERS.items()
        ]

    async def clear(self, folder: str) -> dict[str, Any]:
        """``{folder, cleared, kept}``: the entries deleted and those kept (jobs
        that have not ended). Raises ``KeyError`` for another folder,
        ``Conflict`` and ``RemovalError`` when something in it is in use."""
        if folder not in FOLDERS:
            raise KeyError(folder)
        kept = 0
        if folder == "sets":
            names = [entry["name"] for entry in await asyncio.to_thread(self.files.list_sets)]
            for name in names:
                await asyncio.to_thread(self.files.delete_set, name)
            cleared = len(names)
        elif folder == "history":
            cleared = len(await asyncio.to_thread(self.files.clear_all_history))
        elif folder == "data":
            for job in self.jobs.jobs.values():
                if job.state != "ended" and (job.details.get("data") or job.details.get("report")):
                    raise Conflict(f"{job.title} reads a dataset: clear it when the job has ended")
            names = [entry["name"] for entry in await asyncio.to_thread(self.files.list_data)]
            for name in names:
                await asyncio.to_thread(self.files.delete_data, name)
            cleared = len(names)
        elif folder == "batches":
            running = self.batches.running
            if running is not None:
                raise Conflict(f"the batch {running.name} is running: stop it first")
            names = [entry["name"] for entry in await asyncio.to_thread(self.batches.list)]
            for name in names:
                await asyncio.to_thread(self.batches.delete, name)
                self.results.forget(name)
                self.files.changed("batches", name, "deleted")
            cleared = len(names)
        elif folder == "models":
            cleared = len(await self.removal.delete_all())
        elif folder == "jobs":
            cleared = 0
            for job in self.jobs.list():
                if job.get("state") != "ended":
                    kept += 1
                    continue
                self.jobs.delete(job["id"])
                cleared += 1
        else:
            cleared = await asyncio.to_thread(self.files.clear_speed)
        return {"folder": folder, "cleared": cleared, "kept": kept}

    def _histories(self) -> list[list[Path]]:
        root = self.files.root / "history"
        return [
            [path for path in _visible(place) if path.is_file() and path.suffix == ".jsonl"]
            for place in (root, root / "datasets")
        ]


def _visible(folder: Path) -> list[Path]:
    try:
        return [path for path in folder.iterdir() if not path.name.startswith(".")]
    except OSError:
        return []
