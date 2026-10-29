#!/usr/bin/env bash
# Build installers for PeepGuard
#   macOS  -> dist/PeepGuard-{version}-mac.dmg
#   Linux  -> dist/peepguard-{version}-linux.deb (if dpkg available)
#            dist/PeepGuard-{version}-linux.AppImage (if appimagetool available)
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(grep 'version =' pyproject.toml | head -1 | cut -d'"' -f2)
echo "Building PeepGuard v$VERSION installer..."

# First build the executables
bash packaging/build.sh

OS=$(uname -s | tr '[:upper:]' '[:lower:]')

if [ "$OS" = "darwin" ]; then
    echo ""
    echo "Creating macOS DMG installer..."

    DMG_NAME="PeepGuard-${VERSION}-mac.dmg"
    DMG_PATH="dist/$DMG_NAME"
    TMP_DIR=$(mktemp -d)

    # Create DMG contents
    cp -r dist/PeepGuard.app "$TMP_DIR/"
    ln -s /Applications "$TMP_DIR/Applications"

    # Create a simple README
    cat > "$TMP_DIR/README.txt" << 'EOF'
PeepGuard - Privacy Guardian

Installation:
  Drag PeepGuard.app to the Applications folder.

First Launch:
  1. Open PeepGuard from Applications
  2. Grant camera permission when prompted
  3. (Optional) Click "Setup" to enroll your face

Usage:
  - The app watches for people looking at your screen
  - Screen blurs automatically when someone is detected
  - Use the UI to adjust settings

For help: https://github.com/peepguard/peepguard
EOF

    # Remove old DMG if exists
    rm -f "$DMG_PATH"

    # Create DMG
    hdiutil create -volname "PeepGuard" -srcfolder "$TMP_DIR" -ov -format UDZO "$DMG_PATH"

    # Cleanup
    rm -rf "$TMP_DIR"

    echo ""
    echo "macOS installer created:"
    ls -lh "$DMG_PATH"

elif [ "$OS" = "linux" ]; then
    echo ""
    echo "Creating Linux installer..."

    # Create .deb package if dpkg-deb is available
    if command -v dpkg-deb &> /dev/null; then
        echo "Building .deb package..."

        DEB_NAME="peepguard_${VERSION}_amd64"
        DEB_DIR="dist/$DEB_NAME"

        rm -rf "$DEB_DIR"
        mkdir -p "$DEB_DIR/DEBIAN"
        mkdir -p "$DEB_DIR/usr/bin"
        mkdir -p "$DEB_DIR/usr/share/applications"
        mkdir -p "$DEB_DIR/usr/share/icons/hicolor/256x256/apps"
        mkdir -p "$DEB_DIR/usr/share/doc/peepguard"

        # Copy binary
        cp dist/peepguard "$DEB_DIR/usr/bin/"
        chmod 755 "$DEB_DIR/usr/bin/peepguard"

        # Create control file
        cat > "$DEB_DIR/DEBIAN/control" << EOF
Package: peepguard
Version: $VERSION
Section: utils
Priority: optional
Architecture: amd64
Depends: libgl1, libglib2.0-0, libsm6, libxrender1, libxext6
Maintainer: PeepGuard <peepguard@example.com>
Description: On-device shoulder-surfing detector
 PeepGuard watches your webcam and blurs your screen when
 someone behind you might be looking at it. All processing
 happens on-device - frames are never stored or transmitted.
EOF

        # Create desktop entry
        cat > "$DEB_DIR/usr/share/applications/peepguard.desktop" << EOF
[Desktop Entry]
Name=PeepGuard
Comment=Privacy Guardian - Shoulder surfing detector
Exec=/usr/bin/peepguard ui
Icon=peepguard
Terminal=false
Type=Application
Categories=Utility;Security;
Keywords=privacy;security;camera;screen;
EOF

        # Create copyright file
        cat > "$DEB_DIR/usr/share/doc/peepguard/copyright" << EOF
PeepGuard - On-device shoulder-surfing detector

Copyright (c) 2024 PeepGuard

This software is provided as-is for privacy protection purposes.
EOF

        # Build the package
        dpkg-deb --build "$DEB_DIR"
        mv "${DEB_DIR}.deb" "dist/peepguard-${VERSION}-linux.deb"
        rm -rf "$DEB_DIR"

        echo "Debian package created:"
        ls -lh "dist/peepguard-${VERSION}-linux.deb"
    fi

    # Create AppImage if appimagetool is available
    if command -v appimagetool &> /dev/null; then
        echo ""
        echo "Building AppImage..."

        APPDIR="dist/PeepGuard.AppDir"
        rm -rf "$APPDIR"
        mkdir -p "$APPDIR/usr/bin"
        mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

        cp dist/peepguard "$APPDIR/usr/bin/"
        chmod 755 "$APPDIR/usr/bin/peepguard"

        # AppRun script
        cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${SELF%/*}
exec "${HERE}/usr/bin/peepguard" "$@"
EOF
        chmod 755 "$APPDIR/AppRun"

        # Desktop file
        cat > "$APPDIR/peepguard.desktop" << EOF
[Desktop Entry]
Name=PeepGuard
Comment=Privacy Guardian
Exec=peepguard ui
Icon=peepguard
Terminal=false
Type=Application
Categories=Utility;Security;
EOF

        ARCH=$(uname -m)
        appimagetool "$APPDIR" "dist/PeepGuard-${VERSION}-linux-${ARCH}.AppImage"
        rm -rf "$APPDIR"

        echo "AppImage created:"
        ls -lh "dist/PeepGuard-${VERSION}-linux-"*.AppImage
    fi

    if ! command -v dpkg-deb &> /dev/null && ! command -v appimagetool &> /dev/null; then
        echo "No packaging tools found. Install dpkg-deb or appimagetool for Linux installers."
        echo "Standalone binary available at: dist/peepguard"
    fi
fi

echo ""
echo "Done! Installers are in the dist/ folder."
