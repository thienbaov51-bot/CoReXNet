"""Reject unresolved symbolic/reparse ancestors before any scoped file IO."""
import os
from pathlib import Path
import shutil
import stat

def safe_path(value):
    path=Path(os.path.abspath(os.fspath(value)))
    for node in (*reversed(path.parents),path):
        try:info=node.lstat()
        except FileNotFoundError:continue
        if stat.S_ISLNK(info.st_mode) or bool(getattr(info,'st_file_attributes',0)&0x400):raise ValueError('symbolic/reparse path component prohibited: '+str(node))
    return path

def safe_tree(value):
    root=safe_path(value)
    if not root.exists():return root
    for directory,names,files in os.walk(root,followlinks=False):
        safe_path(directory)
        for name in names+files:safe_path(Path(directory)/name)
    return root

def copy_exact(source,destination):
    source=safe_path(source);destination=safe_path(destination)
    if destination.exists():
        import hashlib
        def digest(path):
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
            return h.digest()
        if digest(source)!=digest(destination):raise FileExistsError('different collision bytes preserved: '+str(destination))
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as incoming,destination.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing)

def copy_tree(source,destination):
    source=safe_tree(source);destination=safe_path(destination)
    if destination.exists():raise FileExistsError('existing target preserved: '+str(destination))
    shutil.copytree(source,destination)

def move_registry(source,destination):
    source=safe_path(source);destination=safe_path(destination)
    if destination.exists():raise FileExistsError('registry collision preserved: '+str(destination))
    source.rename(destination)
