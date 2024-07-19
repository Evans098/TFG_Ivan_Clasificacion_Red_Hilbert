gcc -Wall -o analizadorISCH2 analizador_IvanVF_32pixeles_HA_7.c -lpcap
cd Capturas_filtradas_pcap_32_resto
# cd capturas_pcap_filtradas_32_pruebas
rm -R capturas_bin
mkdir capturas_bin
for i in `ls`
do
dirname=${i%%.*} # quitar el .pcap
mkdir capturas_bin/$dirname
mv $i capturas_bin/$dirname

cd capturas_bin/$dirname
./../../../analizadorISCH2 -f $i #&>/dev/null
# mv capturas_pcap_filtradas/*.bin capturas_pcap_filtradas/capturas_bin
cd ..
cd ..
mv capturas_bin/$dirname/$i .

echo $i analizado y movido a carpeta $dirname
done
cd ..
rm -d Capturas_filtradas_pcap_32_resto/capturas_bin/capturas_bin
# rm -d capturas_pcap_filtradas_32_pruebas/capturas_bin/capturas_bin



