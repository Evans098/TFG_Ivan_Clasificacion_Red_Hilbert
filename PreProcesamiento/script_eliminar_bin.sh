echo empezando a eliminar bins y 1D
# cd capturas_pcap_filtradas_32_HFP/capturas_bin
# cd capturas_pcap_filtradas/bin
cd capturas_pcap_filtradas_32_completas/capturas_bin
for i in `ls`
do
	cd $i

	for j in `ls`
	do
		cd $j
		echo $j
		rm *.bin
		echo ficheros de $j eliminados
		cd ..

	done

	echo TODOS los ficheros de $i eliminados

	cd ..

done
cd ..
cd ..
echo ficheros bin eliminados