# -*- coding: utf-8 -*-
from __future__ import annotations
import base64
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Iterator
import errno
import platform
import tempfile
import ctypes

import yaml
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

MAGIC = b"DSMF1\n"
TAG_LEN = 16
CHUNK = 1024 * 1024  # 1 MiB

# memfd flags（不同 Python/内核的兼容）
_MFD_CLOEXEC = getattr(os, "MFD_CLOEXEC", 0x0001)
_MFD_ALLOW_SEALING = getattr(os, "MFD_ALLOW_SEALING", 0x0002)
try:
    import fcntl

    F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
    F_SEAL_SEAL = 0x0001
    F_SEAL_SHRINK = 0x0002
    F_SEAL_GROW = 0x0004
    F_SEAL_WRITE = 0x0008
    _HAS_SEAL = True
except Exception:
    _HAS_SEAL = False

# ---------- 内部工具 ----------


def _read_header(fin) -> tuple[dict, int, bytes, int]:
    magic = fin.read(len(MAGIC))
    if magic != MAGIC:
        raise ValueError("Not a DSMF1 file or corrupted header.")
    line = fin.readline()
    meta = json.loads(line.decode("utf-8"))
    tag_pos = fin.tell()
    tag = fin.read(TAG_LEN)
    if len(tag) != TAG_LEN:
        raise ValueError("Truncated header (tag).")
    ciph_off = fin.tell()
    return meta, tag_pos, tag, ciph_off


class _DecryptReader(io.RawIOBase):
    """把密文文件对象包装成解密后的只读流（AES-256-GCM）"""

    def __init__(self, fin, key: bytes, nonce: bytes, tag: bytes):
        self._fin = fin
        self._dec = Cipher(
            algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend()
        ).decryptor()
        self._buf = b""
        self._eof = False

    def readable(self):
        return True

    def read(self, n=-1):
        if self._eof and not self._buf:
            return b""
        if n is None or n < 0:
            chunks = [self._buf]
            self._buf = b""
            while True:
                c = self._fin.read(CHUNK)
                if not c:
                    break
                chunks.append(self._dec.update(c))
            self._dec.finalize()
            self._eof = True
            return b"".join(chunks)
        out = bytearray()
        if self._buf:
            take = min(len(self._buf), n)
            out += self._buf[:take]
            self._buf = self._buf[take:]
            n -= take
            if n == 0:
                return bytes(out)
        while n > 0:
            c = self._fin.read(CHUNK)
            if not c:
                self._dec.finalize()
                self._eof = True
                return bytes(out)
            pt = self._dec.update(c)
            if not pt:
                continue
            if len(pt) <= n:
                out += pt
                n -= len(pt)
            else:
                out += pt[:n]
                self._buf = pt[n:]
                n = 0
        return bytes(out)


def _syscall_memfd_create(name: str, flags: int) -> int:
    """
    直接调用内核 syscall 的 memfd_create；在 Python 没有 os.memfd_create 时使用。
    返回 fd；失败抛 OSError(ENOSYS/…)
    """
    # 不同架构的 SYS_memfd_create 号（常见平台）
    SYSNO = {
        "x86_64": 319,
        "aarch64": 279,
        "armv7l": 385,  # 32-bit ARM（armhf）
        "armv6l": 385,
        "i686": 356,  # 32-bit x86
        "ppc64le": 360,
        "riscv64": 279,
    }
    arch = platform.machine()
    sysno = SYSNO.get(arch)
    if sysno is None:
        raise OSError(
            errno.ENOSYS, f"memfd_create syscall number unknown for arch {arch}"
        )

    libc = ctypes.CDLL(None, use_errno=True)
    # long syscall(long number, ...);
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    syscall.argtypes = (ctypes.c_long, ctypes.c_char_p, ctypes.c_uint)

    name_b = name.encode("utf-8")
    fd = syscall(sysno, name_b, ctypes.c_uint(flags))
    if fd < 0:
        e = ctypes.get_errno()
        raise OSError(e, f"memfd_create syscall failed: errno={e}")
    return int(fd)


def _mk_memfd(name: str) -> int:
    """
    首选 memfd；失败则用 /dev/shm 匿名 tmp 文件（unlink 后仅 fd 可见）。
    """
    flags = _MFD_CLOEXEC | _MFD_ALLOW_SEALING

    # 1) Python 自带 API（若存在）
    if hasattr(os, "memfd_create"):
        try:
            return os.memfd_create(name, flags=flags)
        except TypeError:
            # 旧 Python 不支持 flags 参数
            return os.memfd_create(name)
        except Exception:
            pass

    # 2) 直接 syscall
    try:
        return _syscall_memfd_create(name, flags)
    except OSError as e:
        # ENOSYS 等情况下进入降级
        if e.errno not in (errno.ENOSYS, errno.EINVAL, errno.EPERM):
            # 其它错误也可以降级，但记录一下更稳妥
            pass

    # 3) /dev/shm 匿名 tmp（unlink 掉；依然只在内存、不可被其他进程扫描）
    #    说明：我们创建 NamedTemporaryFile(delete=False)，open 后立刻 unlink，
    #    返回其 fd。这样只有持有 fd 的进程能访问，路径已消失。
    shm_dir = Path("/dev/shm")
    shm_dir.mkdir(exist_ok=True)  # 通常已存在
    f = tempfile.NamedTemporaryFile(prefix="dslock_", dir=str(shm_dir), delete=False)
    try:
        fd = f.fileno()
        # 立刻取消目录项，避免被别的进程通过路径探测
        os.unlink(f.name)
    finally:
        # 不关闭文件对象，以免 fd 被关闭；让调用方接管
        pass
    # 设定 CLOEXEC 语义：subprocess 只会继承 pass_fds 的那些
    try:
        import fcntl

        flags_fd = fcntl.fcntl(fd, fcntl.F_GETFD)
        fcntl.fcntl(fd, fcntl.F_SETFD, flags_fd | fcntl.FD_CLOEXEC)
    except Exception:
        pass
    return fd


def _seal_rdonly(fd: int):
    if not _HAS_SEAL:
        return
    try:
        fcntl.fcntl(
            fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL
        )
    except Exception:
        pass


# ---------- YAML 读取 ----------


@dataclass
class KeyConfig:
    key_path: Optional[Path] = None  # 优先：在 YAML 写 key_path: /path/to/dataset.key
    key_b64: Optional[str] = None  # 或者：key_b64: "base64密钥"


def load_key_from_yaml(yaml_path: Path) -> bytes:
    """
    支持：
      1) yaml_path -> YAML 文件，含 key_path: /path/to/dataset.key  或  key_b64: "...."
      2) yaml_path -> 纯 .key 文件（内容是 base64 的 32 字节）
      3) 路径里含 ~ 会自动展开
    返回：原始 32 字节 key
    """
    # 先展开 ~，再转绝对路径（即便文件不存在也能得到规范路径）
    p = Path(os.path.expanduser(str(yaml_path))).resolve(strict=False)

    # 读取文本（YAML 或 .key 都读得通；.key 不是 UTF-8 也没关系）
    try:
        text = p.read_text(encoding="utf-8")
        is_text = True
    except UnicodeDecodeError:
        # 如果是纯二进制（少见），按二进制再试
        raw = Path(p).read_bytes()
        try:
            key = base64.b64decode(raw)
        except Exception as e:
            raise ValueError(
                f"{p} is not valid base64-encoded key and not a YAML file"
            ) from e
        if len(key) != 32:
            raise ValueError(f"{p} decoded length = {len(key)}; need 32 bytes")
        return key
    except FileNotFoundError:
        raise FileNotFoundError(f"Key config not found: {p}")

    data = None
    if is_text:
        try:
            data = yaml.safe_load(text)
        except Exception:
            data = None

    # 情况1：是 YAML 映射，优先读取 key_path / key_b64
    if isinstance(data, dict):
        if "key_path" in data and data["key_path"]:
            kp = Path(os.path.expanduser(str(data["key_path"]))).resolve(strict=False)
            try:
                raw = base64.b64decode(Path(kp).read_bytes())
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"key_path not found: {kp}, maybe you need to get a key file first."
                )
            except Exception as e:
                raise ValueError(f"key_path {kp} is not base64-encoded key") from e
            if len(raw) != 32:
                raise ValueError(
                    f"key_path {kp} decoded length = {len(raw)}; need 32 bytes"
                )
            return raw

        if "key_b64" in data and data["key_b64"]:
            try:
                raw = base64.b64decode(str(data["key_b64"]))
            except Exception as e:
                raise ValueError("key_b64 is not valid base64") from e
            if len(raw) != 32:
                raise ValueError(f"key_b64 decoded length = {len(raw)}; need 32 bytes")
            return raw

        raise ValueError("YAML must contain 'key_path' or 'key_b64'.")

    # 情况2：不是 YAML，当作 .key 文本（base64 32B）
    try:
        raw = base64.b64decode(text.strip())
    except Exception as e:
        raise ValueError(f"{p} is neither YAML nor base64-encoded key") from e
    if len(raw) != 32:
        raise ValueError(f"{p} decoded length = {len(raw)}; need 32 bytes")
    return raw


# ---------- 解密 API ----------


def decrypt_enc_to_memfds(enc_file: Path, key_bytes: bytes) -> Dict[str, int]:
    """
    把 .enc 文件（DSMF1）解密为 memfd，返回 {归档内相对路径: fd}
    注意：调用方负责在使用完后关闭这些 fd（或使用下面的上下文管理器）。
    """
    fd_map: Dict[str, int] = {}
    with open(enc_file, "rb") as fin:
        meta, _, tag, ciph_off = _read_header(fin)
        if meta.get("kdf") == "scrypt":
            raise ValueError(
                "This helper expects raw key mode. Please use raw key (not passphrase)."
            )
        nonce = base64.b64decode(meta["nonce_b64"])
        fin.seek(ciph_off, os.SEEK_SET)
        # 为稳妥起见，先把明文 tar 全读到内存（保持简单可靠；如需超大文件可改为流式解析）
        rdr = _DecryptReader(fin, key_bytes, nonce, tag)
        plaintext = bytearray()
        while True:
            chunk = rdr.read(CHUNK)
            if not chunk:
                break
            plaintext.extend(chunk)
    tf = tarfile.open(mode="r:", fileobj=io.BytesIO(bytes(plaintext)))
    for m in tf.getmembers():
        if not m.isfile():
            continue
        fobj = tf.extractfile(m)
        if fobj is None:
            continue
        fd = _mk_memfd(m.name.replace("/", "_")[:200] or "file")
        with os.fdopen(fd, "wb", closefd=False) as out:
            remaining = m.size
            while remaining > 0:
                b = fobj.read(min(CHUNK, remaining))
                if not b:
                    break
                out.write(b)
                remaining -= len(b)
            out.flush()
        _seal_rdonly(fd)
        fd_map[m.name] = fd
    return fd_map


@dataclass
class MemfdDataset:
    """上下文管理器：自动清理 memfd。"""

    enc_file: Path
    key_yaml: Path

    _fd_map: Optional[Dict[str, int]] = None

    def __enter__(self) -> "MemfdDataset":
        key = load_key_from_yaml(self.key_yaml)
        self._fd_map = decrypt_enc_to_memfds(self.enc_file, key)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd_map:
            for fd in self._fd_map.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._fd_map = None

    # ---- 便捷方法 ----
    @property
    def files(self) -> Dict[str, int]:
        if self._fd_map is None:
            raise RuntimeError(
                "Dataset not opened. Use 'with MemfdDataset(...) as ds:'"
            )
        return self._fd_map

    def open_fd(self, relpath: str, mode: str = "rb"):
        """返回可读文件对象，底层指向 /proc/self/fd/<fd>（只读）"""
        fd = self.files[relpath]
        return open(f"/proc/self/fd/{fd}", mode)

    def iter_paths(self) -> Iterator[str]:
        return iter(self.files.keys())
