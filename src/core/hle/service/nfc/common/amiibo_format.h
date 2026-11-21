// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/hle/service/nfp/nfp_types.h"

// Layout helpers for amiibo dumps. n2den has no amiibo decryption: amiibo files are used as
// plain (already decrypted) data.
namespace Service::NFP::AmiiboFormat {

/// Validates that the amiibo file is not corrupted
bool IsAmiiboValid(const EncryptedNTAG215File& ntag_file);

/// Validates that the amiibo file is not corrupted
bool IsAmiiboValid(const NTAG215File& ntag_file);

/// Converts from the on-tag byte order to the plain data layout
NTAG215File NfcDataToEncodedData(const EncryptedNTAG215File& nfc_data);

/// Converts from the plain data layout to the on-tag byte order
EncryptedNTAG215File EncodedDataToNfcData(const NTAG215File& encoded_data);

} // namespace Service::NFP::AmiiboFormat
