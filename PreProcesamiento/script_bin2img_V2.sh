echo empezando a convertir texto a imagenes...
cd Capturas_filtradas_pcap_32_resto/capturas_bin
# cd capturas_pcap_filtradas/bin
for i in `ls`
do
	cd $i
	
	for j in `ls`
	do
		# SCRIPT=$(readlink -f $0);
		# dir_base=`dirname $SCRIPT`;
		# echo $dir_base
		# echo $j
		python3 ./../../../binary2image.py $j
		echo carpetas $j procesada correctamente

	done

	echo paquetes $i convertidos a imagenes 1D y 2D

	cd ..

done
cd ..
cd ..
echo paquetes convertidos a imagenes 1D y 2D

