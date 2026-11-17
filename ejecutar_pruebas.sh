#!/usr/bin/env bash
# Ejecuta todas las pruebas (datos sintéticos). Uso:  bash ejecutar_pruebas.sh
set -e
cd "$(dirname "$0")"
for t in t_procesado t_vectores t_morfologia t_copas t_estructuras t_pipeline; do echo "== $t"; python pruebas/$t.py | tail -3; done
echo "== figuras"; python pruebas/t_figuras.py /tmp/forestmap_figs | tail -2
echo "== exportar"; python pruebas/t_exportar.py /tmp/forestmap_paquete | tail -1
echo "TODO OK"
