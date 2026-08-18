# Emerald Kaizo ROM Fingerprint

This is a local fingerprint of the ROM found in Downloads. The ROM itself is
not copied into this repository.

## Local File

- Path: `/Users/arc/Downloads/kaizo-emerald.gba`
- Size: `16,777,216` bytes
- GBA title header: `POKEMON EMER`
- Game code: `BPEE01`
- SHA-256: `ceb218cf629343dff7fc284f8c2de246fbc7f3e817a24f8b096b8d7102cbac4b`
- MD5: `b48a5df9dfa88ce7cf0465bee8bbff94`
- SHA-1: `a7c4e34fafb53f2d5283eb6d43dad285c0dd40a8`
- CRC32: `66c1cce7`

The macOS download metadata points to a third-party itch.io reupload. The
header contains no Kaizo version marker, so this file cannot be identified as
v2.1 from the header alone.

## Base ROM Verification

The adjacent vanilla ROM, `Pokemon - Emerald Version (USA, Europe).gba`,
matches the source checksums published by ROMHacking.net for the v2.1 hack:

- MD5: `605b89b67018abcea91e693a4dd25be3`
- SHA-1: `f3ae088181bf583e55daf962a92bb46f4f1d07b7`
- CRC32: `1f1c08fb`

The public `Emerald Kaizo 1.1.bps` patch from the SHF-Kaizo-Patches repository
was applied to that verified base in a temporary directory. Its output did
not match the downloaded Kaizo ROM. This is expected if the downloaded file is
the older published v2.1 build, a later build, or a repack, but it means the
version is currently unresolved.

ROMHacking.net exposes the v2.1 download at:

`https://www.romhacking.net/download/hacks/4291/7608a162aef41887055f9dddc8044b7ab6e431b9f51d150a32f45e61018cd493`

A direct read of that endpoint returned HTTP 403 during this session, so no
official v2.1 output hash was obtained.

## Analysis Consequence

The generic IV report uses the documented Gen III marginal model. It must not
claim that every Kaizo-specific encounter table, EV patch, or wild-generation
change has been verified in this particular ROM until the binary is identified
or inspected directly.
