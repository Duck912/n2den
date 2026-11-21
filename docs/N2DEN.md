# n2den

n2den is a Switch 2 recompulator. It continues an imported v1.6.7.2 source
snapshot of Eden. The base project and its contributors are credited in
[README.md](../README.md#credits).

## Source provenance

- Only the emulator source directory was imported from the release's `source.zip`.
- Archive comment: `8882e6542dc3334e205f3421d03ee6a6438a017c` (archive-provided
  provenance; this repository does not contain that commit's history).
- Source ZIP SHA-256: `feff31d3eb16f0e4e40008288db7a11f9ca9a031e26773eccb945df8535c00bc`.
- The initial import was checked against 4,857 source files in the archive.
  Executable modes were restored from ZIP attributes.
- Original licenses, notices and contributor credits are retained.
- No games, keys, system files, saves, runtime caches, vendor DLLs or portable user data
  are included in this repository.

## Ocarina of Time startup fix

Ported **upstream PR #567**, by **zirulo**, "kernel: Fix OOT startup by
implementing WaitIfEqual64". Original commit:
`b6706b0c29f876d0c96a56563080fce494721337`.

The patch accepts address-arbitration type 3, retains the full AArch64 comparison
value, reads an aligned 64-bit guest value, and reuses the existing wait/signal
queue. Existing 32-bit arbitration modes and the AArch32 ABI remain unchanged.
It addresses the early `InvalidEnumValue` / `SetTerminateResult 2001-0120` abort.
On its own it does not establish shader correctness or gameplay compatibility.

Run the generator ABI regression checks from the repository root:

```text
python tools/tests/test_wait_for_address_abi.py -v
```

The AArch64 test failed before the patch (int32_t instead of int64_t) and passes
after it. The AArch32 wrapper-layout test also passes.

## Builds and updates

Release builds use MSVC.
Keep test builds and their user data separate from other installs.
