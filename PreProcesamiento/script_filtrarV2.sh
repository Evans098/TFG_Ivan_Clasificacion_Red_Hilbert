mkdir hangouts
for i in `ls ../../sda`
do
tshark -r ../../sda/$i -Y "tcp && tcp.len != 0 || ssl || gquic || data " -w hangouts/f_$i
echo "$i filtrado y copiado"
done
