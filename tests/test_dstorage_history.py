"""Local sidecar generation and resume from a hand-built RDS5 pool."""
from pathlib import Path
from array import array
from contextlib import redirect_stdout
import io
import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"tools"), str(ROOT/"tests")]
from test_bcpack_history_calib import (_encode_bc6h_mode14_tiled, _encode_bc7_mode6_tiled, _pack_etpf,
                                       _write_custom_rds5_pool)
import build_dstorage_history as history


def _bc7_mip(mip, scale, half):
    """Mode-6 BC7 blocks with a gradient, so a sibling mip predicts them."""
    buf = bytearray()
    for by in range(mip["blocks_h"]):
        for bx in range(mip["blocks_w"]):
            r = min(254, (32 + bx * 6 * scale + half) & 0xFE)
            g = min(254, (80 + by * 6 * scale + half) & 0xFE)
            acc, pos = 1 << 6, 7
            for v in (r >> 1, r >> 1, g >> 1, g >> 1, 70, 70, 127, 127):
                acc |= (v & 0x7F) << pos
                pos += 7
            acc |= (r & 1) << pos | (r & 1) << (pos + 1)
            buf += acc.to_bytes(16, "little")
    return bytes(buf)


def _bc6h_mip(mip, scale, half):
    """Mode-14 BC6H blocks with a gradient."""
    plan, _ = history.calib.bcpack._bc6h_plan(0x0F)
    buf = bytearray()
    for by in range(mip["blocks_h"]):
        for bx in range(mip["blocks_w"]):
            f = [[w, 0, 0, 0] for w in (0x3800 + bx * 16 * scale + half, 0x3A00 + by * 16 * scale + half, 0x3C00)]
            acc = 0x0F
            for c, q, bit, pos in plan:
                if c >= 0 and (f[c][q] >> bit) & 1:
                    acc |= 1 << pos
            buf += acc.to_bytes(16, "little")
    return bytes(buf)


def _synthetic_pools(folder, count):
    """`count` pools of hand-built BC6H and BC7 textures (a BCPack top mip over a zlib mip), plus after every third
    one a pool whose packed BC7 tail only a decoded BCPack sibling references. History entries repeat across
    pools with different reference counts, so the capped observation lists fill up over several pools. Sizes
    differ, so the largest-first submission order is not the pool order."""
    ttpf, calib = history.ttpf, history.calib
    solid = bytes([0x40]) + bytes(15)
    paths = []
    for n in range(count):
        size = (64, 128, 64, 256)[n % 4]
        ref = (8, 40, 120, 250)[n % 4]
        lay7 = ttpf.surface_layout(size, size, 98, 2, ttpf.tile_bytes(4))
        lay6 = ttpf.surface_layout(size, size, 95, 2, ttpf.tile_bytes(4))
        t7 = [calib.retile_mip(_bc7_mip(m, i + 1, 3 * i + n), lay7, m) for i, m in enumerate(lay7["mips"])]
        t6 = [calib.retile_mip(_bc6h_mip(m, i + 1, 8 * i + n), lay6, m) for i, m in enumerate(lay6["mips"])]
        ep7 = tuple([0x40 + 8 * (n % 3), 0x60, 0x90, 0xFE] * 6)
        eh6 = tuple((0x1800 + 0x100 * (n % 3) + 0x800 * c, 0x200, 0x20, 0x20) for c in range(3))
        top7 = _encode_bc7_mode6_tiled(t7[0], secret_entry=91 + n % 2, secret_ep=ep7, secret_ix=(0,) * 32,
                                       ref_count=ref)
        top6 = _encode_bc6h_mode14_tiled(t6[0], secret_entry=55 + n % 2, secret_eh=eh6, secret_ih=(0,) * 16,
                                         ref_count=ref)
        base = ttpf.align(16 + 2 * (24 + 32), 4096)
        r7, r6 = lay7["mips"][0]["region"], lay6["mips"][0]["region"]
        z7, z6 = lay7["mips"][1]["region"], lay6["mips"][1]["region"]
        records = [_pack_etpf(size, size, 1, 1, 98, base), _pack_etpf(size, size, 1, 1, 95, base + z7 + r7)]
        chunks = [(8, z7, zlib.compress(t7[1], 9)), (7, r7, top7), (8, z6, zlib.compress(t6[1], 9)), (6, r6, top6)]
        path = Path(folder)/f"synthetic-{n:02d}.ttpf"
        _write_custom_rds5_pool(path, records, chunks)
        paths.append(str(path))
        if n % 3 == 0:
            # The packed tail of test_packed_tail_fits_against_decoded_bcpack_sibling below.
            tail = _encode_bc7_mode6_tiled(solid * 256, secret_entry=255, secret_ep=(0, 0, 0, 128) * 6,
                                           ref_count=255)
            path = Path(folder)/f"synthetic-{n:02d}-tail.ttpf"
            _write_custom_rds5_pool(path, [_pack_etpf(128, 128, 4, 1, 98, 4096)],
                                    [(7, 4096, tail), (7, 4096, _encode_bc7_mode6_tiled(solid * 256)),
                                     (8, 16384, zlib.compress(solid * 1024))])
            paths.append(str(path))
        if n % 3 == 1:
            # The late 2D mip of test_visible_2d_mip_starts_after_first_256_compressed_blocks, for fit_slices.
            alpha = 128 - 8 * (n % 2)
            late = _encode_bc7_mode6_tiled(solid * 512, secret_entry=255, secret_ep=(0, 0, 0, alpha) * 6,
                                           ref_count=255)
            path = Path(folder)/f"synthetic-{n:02d}-late.ttpf"
            _write_custom_rds5_pool(path, [_pack_etpf(128, 128, 4, 1, 98, 4096)],
                                    [(7, 8192, late), (8, 16384, zlib.compress(solid * 1024))])
            paths.append(str(path))
    return paths


class RuntimeHistoryTest(unittest.TestCase):
    def test_constraints_preserve_confirmed_bits(self):
        docs = {"bc6h": history.calib.Bc6hCalibrator().build_document()[0],
                "bc7": history.calib.Bc7Calibrator().build_document()[0]}
        for name in ("bc6h", "bc7"):
            row = docs[name]
            values, confirmed = (("indexHistory", "indexConfirmedBits") if name == "bc6h" else
                                 ("historyIndices", "historyIndexConfirmedBits"))
            row[values][8][5] = 12
            row[confirmed][8][5] = 2
        votes = array("Q", [0]) * (2*256*35*16)
        for base in (0, 256*35*16):
            start = base + (8*32+5)*16
            # Only candidates 3 and 14 satisfy the anchor. The known top two bits select 14.
            for v in range(16):
                votes[start+v] = int(v not in (3, 14))
        with tempfile.TemporaryDirectory() as name:
            model = Path(name)/"history.json"
            model.write_text(json.dumps(docs))
            history.refine(docs, votes, model)
        self.assertEqual(docs["bc6h"]["indexHistory"][8][5], 14)
        self.assertEqual(docs["bc7"]["historyIndices"][8][5], 14)
        self.assertEqual(docs["bc7"]["historyIndexConfirmedBits"][8][5], 2)

    def test_shared_red_parity_preserves_partial_endpoint_precision(self):
        docs = {"bc6h": history.calib.Bc6hCalibrator().build_document()[0],
                "bc7": history.calib.Bc7Calibrator().build_document()[0]}
        bc7 = docs["bc7"]
        for field in (0, 4):
            bc7["historyEndpoints"][9][field] = 96
            bc7["historyEndpointBits"][9][field] = 5
            bc7["historyEndpointConfirmedBits"][9][field] = 4
        votes = array("Q", [0]) * (2*256*35*16)
        start = (256*35 + 256*32 + 9*3)*16
        for pair in range(16):
            votes[start + pair] = int(not ((pair // 4 ^ pair % 4) & 2))
        with tempfile.TemporaryDirectory() as name:
            model = Path(name)/"profile.json"
            model.write_text(json.dumps(docs))
            history.refine(docs, votes, model)
        loaded = history.bptc.load_bc7(docs)
        a, b = (loaded["history_endpoints"][9][p] for p in (0, 4))
        self.assertEqual((a ^ b) & 2, 2)
        for field in (0, 4):
            self.assertEqual(loaded["history_endpoints"][9][field] >> 3, 96 >> 3)
            self.assertEqual(loaded["history_endpoint_bits"][9][field], 5)
            self.assertEqual(loaded["history_endpoint_confirmed_bits"][9][field], 4)
        bc7["historyEndpointBits"][9][0] = 7
        for field in (0, 1):
            # Field zero's known prefix now includes bit 1. Field one is green.
            bc7["historyEndpointLow2Fits"] = [[9, field, 2]]
            with self.assertRaises(ValueError):
                history.bptc.load_bc7(docs)

    def test_visible_2d_mip_starts_after_first_256_compressed_blocks(self):
        # Mip 1 begins 4 KiB into the compressed tail; its larger sibling is raw BC7.
        # Carry the synthetic pre-start alpha value to blocks 256..258.
        solid = bytes([0x40]) + bytes(15)
        secret = (0, 0, 0, 128) * 6
        tail = _encode_bc7_mode6_tiled(solid * 512, secret_entry=255, secret_ep=secret, ref_count=255)
        with tempfile.TemporaryDirectory() as name:
            pool = Path(name)/"late-mip.ttpf"
            record = _pack_etpf(128, 128, 4, 1, 98, 4096)
            _write_custom_rds5_pool(pool, [record], [(7, 8192, tail), (8, 16384, zlib.compress(solid * 1024))])
            bc7 = history.fit_slices([pool])["bc7"]
        for field in (3, 7):
            self.assertGreaterEqual(bc7["historyEndpointBits"][255][field], 4)
            self.assertEqual(bc7["historyEndpointConfirmedBits"][255][field], 0)
            self.assertLessEqual(abs(bc7["historyEndpoints"][255][field] - 128), 4)

    def test_packed_tail_fits_against_decoded_bcpack_sibling(self):
        # Mip 0 is zlib, mip 1 a self-contained BCPack stream and the packed tail (mips 2-4) starts its
        # chains at history entry 255, whose alpha no raw or zlib sibling of the tail can show.
        solid = bytes([0x40]) + bytes(15)
        secret = (0, 0, 0, 128) * 6
        tail = _encode_bc7_mode6_tiled(solid * 256, secret_entry=255, secret_ep=secret, ref_count=255)
        with tempfile.TemporaryDirectory() as name:
            pool = Path(name)/"bcpack-parent.ttpf"
            record = _pack_etpf(128, 128, 4, 1, 98, 4096)
            _write_custom_rds5_pool(pool, [record], [(7, 4096, tail), (7, 4096, _encode_bc7_mode6_tiled(solid * 256)),
                                                     (8, 16384, zlib.compress(solid * 1024))])
            fits = {6: history.calib.Bc6hCalibrator(), 7: history.calib.Bc7Calibrator()}
            history.calib.calibrate_from_pools([str(pool)], fits)
            docs = {history.calib.CODEC_KEYS[c]: f.build_document()[0] for c, f in fits.items()}
            history.extend_missing(docs, history.fit_slices([pool]))
            bcpack = history.calib.bcpack
            with self.assertRaises(bcpack.BcpackProvisionalError):
                bcpack.decode(tail, 7, 4096, allow_provisional=True, history=docs)
            history.fit_decoded_neighbors([pool], docs)
        bc7 = docs["bc7"]
        for field in (3, 7):
            self.assertGreaterEqual(bc7["historyEndpointBits"][255][field], 4)
            self.assertEqual(bc7["historyEndpointConfirmedBits"][255][field], 0)
            self.assertLessEqual(abs(bc7["historyEndpoints"][255][field] - 128), 4)
        self.assertEqual(bcpack.decode(tail, 7, 4096, allow_provisional=True, history=docs), solid * 256)

    def test_packed_tail_fits_against_its_own_refused_stream(self):
        # One BCPack stream holds the packed tail (blocks 0-255) and mip 1 (blocks 256-511). Chains from history
        # entry 255 stay inside the tail, so the stream is refused and no fit reaches the entry, but mip 1 is
        # determined and is the reference.
        solid = bytes([0x40]) + bytes(15)
        secret = (0, 0, 0, 128) * 6
        stream = _encode_bc7_mode6_tiled(solid * 512, secret_entry=255, secret_ep=secret, ref_count=200)
        with tempfile.TemporaryDirectory() as name:
            pool = Path(name)/"one-stream.ttpf"
            record = _pack_etpf(128, 128, 4, 1, 98, 4096)
            _write_custom_rds5_pool(pool, [record], [(7, 8192, stream), (8, 16384, zlib.compress(solid * 1024))])
            fits = {6: history.calib.Bc6hCalibrator(), 7: history.calib.Bc7Calibrator()}
            history.calib.calibrate_from_pools([str(pool)], fits)
            docs = {history.calib.CODEC_KEYS[c]: f.build_document()[0] for c, f in fits.items()}
            history.extend_missing(docs, history.fit_slices([pool]))
            bcpack = history.calib.bcpack
            with self.assertRaises(bcpack.BcpackProvisionalError):
                bcpack.decode(stream, 7, 8192, allow_provisional=True, history=docs)
            history.fit_decoded_neighbors([pool], docs)
        bc7 = docs["bc7"]
        for field in (3, 7):
            self.assertGreaterEqual(bc7["historyEndpointBits"][255][field], 4)
            self.assertEqual(bc7["historyEndpointConfirmedBits"][255][field], 0)
            self.assertLessEqual(abs(bc7["historyEndpoints"][255][field] - 128), 4)
        self.assertEqual(bcpack.decode(stream, 7, 8192, allow_provisional=True, history=docs), solid * 512)

    def test_build_resume_and_repair(self):
        trainer = os.environ.get("LEONIDA_TEST_BCPACK_TRAINER")
        if not trainer and not shutil.which("cl" if os.name == "nt" else "c++"):
            self.skipTest("a host C++ compiler or prebuilt fitter is required")
        with tempfile.TemporaryDirectory() as name:
            work = Path(name)
            pool = work/"synthetic.ttpf"
            # Solid mode-6 BC7 blocks. No copyrighted payload or title files.
            blocks = (bytes([0x40]) + bytes(15)) * 256
            encoded = _encode_bc7_mode6_tiled(blocks)
            record = _pack_etpf(64, 64, 1, 1, 98, 4096)
            _write_custom_rds5_pool(pool, [record], [(7, 4096, encoded)])
            sidecar = work/"shaders"/"bcpack"/"history.bcph"
            def build():
                history.build(pool, sidecar, Path(trainer) if trainer else None, if_needed=True)
            build()
            original = sidecar.read_bytes()
            self.assertEqual((original[:8], len(original)), (b"BCPHIST1", 60936))
            when = sidecar.stat().st_mtime_ns
            build()
            self.assertEqual(when, sidecar.stat().st_mtime_ns)
            sidecar.write_bytes(bytes(len(original)))
            build()
            self.assertEqual(original, sidecar.read_bytes())


class _InlineFuture:
    def __init__(self, executor, task, args):
        self.executor, self.task, self.args, self.value = executor, task, args, None

    def result(self):
        self.executor.drain()
        return self.value


class _ReversedExecutor:
    """Runs submitted tasks in this process, the last submitted first, once a result is asked for. A result goes
    through pickle, as a worker's does."""
    def __init__(self):
        self.pending, self.finished = [], []

    def submit(self, task, *args):
        future = _InlineFuture(self, task, args)
        self.pending.append(future)
        return future

    def drain(self):
        while self.pending:
            future = self.pending.pop()
            future.value = pickle.loads(pickle.dumps(future.task(*future.args)))
            self.finished.append(future.args[0])


def _fit_all(pools, executor):
    docs = history.calibrate(pools, executor)
    history.extend_missing(docs, history.fit_slices(pools, executor))
    history.fit_decoded_neighbors(pools, docs, executor=executor)
    return json.dumps(docs)


class ParallelHistoryTest(unittest.TestCase):
    """--jobs N fits the profile of --jobs 1: worker results are merged in pool order, never in arrival order."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        _synthetic_pools(cls.tmp.name, 7)
        cls.pools = history.calib._resolve_pools([cls.tmp.name])
        cls.sequential = _fit_all(cls.pools, None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixture_fills_the_caps_over_several_pools_and_depends_on_order(self):
        fits = history._fits()
        history.calib.calibrate_from_pools(self.pools, fits)
        first = history._fits()
        history.calib.calibrate_from_pools(self.pools[:1], first)
        for codec, name, cap in ((6, "w_obs", 64), (6, "d_obs", 96), (7, "ep_obs", 60)):
            full = [key for key, obs in getattr(fits[codec], name).items() if len(obs) == cap]
            self.assertTrue(full, name)
            # Some list only reaches its cap with the later pools, so the merge order decides what it keeps.
            self.assertTrue(any(len(getattr(first[codec], name).get(key, ())) < cap for key in full), name)
        self.assertNotEqual(_fit_all(self.pools[::-1], None), self.sequential)

    def test_workers_finishing_out_of_order_merge_like_one_process(self):
        executor = _ReversedExecutor()
        self.assertEqual(_fit_all(self.pools, executor), self.sequential)
        self.assertEqual(set(executor.finished), set(self.pools))
        self.assertNotEqual(executor.finished[:len(self.pools)], self.pools)

    def test_spawned_workers_merge_like_one_process(self):
        with history._executor(4, self.pools) as executor:
            self.assertIsNotNone(executor)
            self.assertEqual(_fit_all(self.pools, executor), self.sequential)

    def test_mask_replay_follows_pool_order(self):
        # The AND is skipped when it would empty the slot, so the masks' order matters: 101 & 110 = 100, and then
        # 100 & 011 would be empty. A worker starting from 0xFFFF ends at 010, which merged as a mask gives 101.
        for cls, p in ((history.calib.Bc6hCalibrator, 5), (history.calib.Bc7Calibrator, 16)):
            total, part = cls(), cls()
            total.valid_ix[3][p] = 0b101
            part.mask_log = {(3, p): [0b110, 0b011]}
            part.valid_ix[3][p] = 0b010
            total.absorb(part)
            self.assertEqual(total.valid_ix[3][p], 0b100)
            with self.assertRaises(ValueError):
                total.absorb(cls())  # no mask log

    def test_build_bytes_do_not_depend_on_jobs(self):
        trainer = os.environ.get("LEONIDA_TEST_BCPACK_TRAINER")
        if not trainer and not shutil.which("cl" if os.name == "nt" else "c++"):
            self.skipTest("a host C++ compiler or prebuilt fitter is required")
        trainer = Path(trainer) if trainer else None
        with tempfile.TemporaryDirectory() as name:
            one, four = Path(name)/"one"/"history.bcph", Path(name)/"four"/"history.bcph"
            with redirect_stdout(io.StringIO()):
                history.build(Path(self.tmp.name), one, trainer, jobs=1)
                history.build(Path(self.tmp.name), four, trainer, jobs=4)
                self.assertEqual(one.read_bytes(), four.read_bytes())
                # The --if-needed stamp does not name the number of jobs either.
                when = one.stat().st_mtime_ns
                history.build(Path(self.tmp.name), one, trainer, if_needed=True, jobs=4)
            self.assertEqual(when, one.stat().st_mtime_ns)
