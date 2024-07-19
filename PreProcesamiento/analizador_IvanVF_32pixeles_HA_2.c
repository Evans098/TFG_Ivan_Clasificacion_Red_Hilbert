/***************************************************************************
 analizador.c

 main y analizar_paquete()

 Compila: gcc -Wall -o analizador analizador.c -lpcap
 Ejecuta: ./analizador -f fichero.pcap
 		  ./analizador -e conexion      ----    NO ES NECESARIO PARA EL TFM

		  sudo mount --bind /mnt/f/Estudiante/TFGF/ ~/Desktop/TFGF/

 Autor: Iván Rengifo De La Cruz

Se parte del código desarrollado por el autor, Ignacio Sotomonte y Victor Morales Gomez.
 
 Y pagina web de filtrado tcp/ip SNIFFEX-1.C

 Cambios:	
 El filtrado de paquetes se escribe directamente desde el terminal.
 Por ejemplo.
 ./disector -f quic.pcap -e "src or dst 192.168.0.1"

 Usaria como filtro src or dst 192.168.0.1 y filtraria los paquetes que tengan
 192.168.0.1 en ip dst o ip src.

 Para la estructura de datos se utiliza una tabla hash.
 El código de partida de esta tabla hash se puede consultar aqui:

 https://www.tutorialspoint.com/data_structures_algorithms/hash_table_program_in_c.htm

  Este código solo es para las clases de Chat e Email

***************************************************************************/

#include "analizador.h"
//#include "hashtable.h"
#include <endian.h>
#include <time.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define SIZE 20000
#define DIM 32
#define SIGNAL_LENGTH 1024

int numpktout = 10000;
int quic_port = 443;


char *token; // nombre del fichero
char name[50];

int contador_flujos = 0;

void d2xy(int n, int d, int *x, int *y);
void rot(int n, int *x, int *y, int rx, int ry);

void horizontal_flip(char signal[], int length, char senial_flip[]);

int get_Np(float alpha);
void permute_sequence(float alpha, char signal[], int length, char signal_permuted[]);

void interpolate_signal(char signal[], int length, char interpolated_signal[]);
void sample_sequence(char signal[], int length, int T, char sampled_signal[]);

int get_Nt(int alpha);
void translate_segment(int alpha, char signal[], int length);

void add_gaussian_noise(char cadena[], int caracteristica, float alpha, float sigma);



typedef struct Flujo {
  int id_f;
}Flujo;

struct Flujo* hashArrayFlujo[SIZE];

int main(int argc, char **argv)
{
	char errbuf[PCAP_ERRBUF_SIZE];
	
	int flag_e = 0;
	int flag_n = 0;
	int long_index = 0, retorno = 0;
	char opt;
	
	/* DECLARACION DE VARIABLES DE FILTRO */
	/* VER SNIFFEX-1.C */

	char filter_exp[] = "";		/* filter expression*/
	struct bpf_program fp;			/* compiled filter program (expression) */
	//bpf_u_int32 mask = 0;			/* subnet mask */
	bpf_u_int32 net = 0;			/* ip */

	// Nombre del fichero a global
    strcpy(name,argv[2] );
	const char s[2] = ".";
	// char * prueba;
	// prueba = strrchr(name,'/');
	// printf("Valor despues / = %s\n",prueba );
	token = strtok(name, s);
	//printf("token=%s\n",token );


	if (signal(SIGINT, handleSignal) == SIG_ERR) {
		//printf("Error: Fallo al capturar la senal SIGINT.\n");
		exit(ERROR);
	}

	if (argc == 1) {
		exit(ERROR);
	}

	static struct option options[] = {
		{"f", required_argument, 0, 'f'},
		{"i",required_argument, 0,'i'},
		{"e", required_argument, 0, 'e'},
		{"n",required_argument,0,'n'},
		{"p",required_argument,0,'p'},
		{"h", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	while ((opt = getopt_long_only(argc, argv, "f:i:e:n:p:h", options, &long_index)) != -1) {
		switch (opt) {
		case 'i' :
			if(descr) { // comprobamos que no se ha abierto ninguna otra interfaz o fichero
				//printf("Ha seleccionado más de una fuente de datos\n");
				pcap_close(descr);
				exit(ERROR);
			}
		
			if ( (descr = pcap_open_live(optarg, 1518, 0, 100, errbuf)) == NULL){
				//printf("Error: pcap_open_live(): Interface: %s, %s %s %d.\n", optarg,errbuf,__FILE__,__LINE__);
				exit(ERROR);
			}
			break;

		case 'f' :
			if(descr) { // comprobamos que no se ha abierto ninguna otra interfaz o fichero
				//printf("Ha seleccionado más de una fuente de datos\n");
				pcap_close(descr);
				exit(ERROR);
			}

			if ((descr = pcap_open_offline(optarg, errbuf)) == NULL) {
				//printf("Error: pcap_open_offline(): File: %s, %s %s %d.\n", optarg, errbuf, __FILE__, __LINE__);
				exit(ERROR);
			}

			break;

		case 'e' :
			//printf("Filtro introducido:%s \n",argv[4]);
			flag_e = 1;						
			break;

		case 'h' :
			printf("Ayuda. Ejecucion: %s <-f traza.pcap / -i eth0> [-e ''filter_exp''] [-n num_pkts_delete] [-p quic port]\n", argv[0]);
			exit(ERROR);
			break;

		case 'n' :
			if(flag_e == 0){
				printf("Please enter a filter before variables:\n");
				printf("Ayuda. Ejecucion: %s <-f traza.pcap / -i eth0> [-e ''filter_exp''] [-n num_pkts_delete] [-p quic port]\n", argv[0]);
				exit(ERROR);
			}
			//printf("Variables introducidas: num pkts delete= %s, quic port= %s\n",argv[6],argv[7]);
			numpktout = atoi(argv[6]);
			flag_n = 1;

			//printf("Num pkt %d\n",numpktout );
			//printf("flag_n %d\n",flag_n );
			//printf("quic port %d\n",quic_port );
			break;

		case 'p' :
			if(flag_e == 0){
				printf("Please enter a filter before variables:\n");
				printf("Ayuda. Ejecucion: %s <-f traza.pcap / -i eth0> [-e ''filter_exp''] [-n num_pkts_delete] [-p quic port]\n", argv[0]);
				exit(ERROR);
			}
			//printf("Variables introducidas: num pkts delete= %s, quic port= %s\n",argv[6],argv[7]);
			if(flag_n == 0)
				quic_port = atoi(argv[6]);
			else
				quic_port = atoi(argv[8]);


			//printf("Num pkt %d\n",numpktout );
			//printf("quic port %d\n",quic_port );
			break;

		case '?' :
		default:
			//printf("Error. Ejecucion: %s <-f traza.pcap / -i eth0> [-e ''filter_exp'']: %d\n", argv[0], argc);
			exit(ERROR);
			break;
		}
	}

	if (!descr) {
		//printf("No selecciono ningún origen de paquetes.\n");
		return ERROR;
	}

	//printf("\n");

	if(argc == 5 && flag_e == 1){
		//printf("Se ha aplicado el filtro anterior.\n");
		if (pcap_compile(descr, &fp, argv[4], 0, net) == -1) {
		fprintf(stderr, "Couldn't parse filter %s: %s\n",
		    filter_exp, pcap_geterr(descr));
		exit(EXIT_FAILURE);
		}
	}	
	else{
		//printf("No se aplica filtro.\n");
		if (pcap_compile(descr, &fp, filter_exp, 0, net) == -1) {
		fprintf(stderr, "Couldn't parse filter %s: %s\n",
		    filter_exp, pcap_geterr(descr));
		exit(EXIT_FAILURE);	
		}
	}

	/* apply the compiled filter */
	if (pcap_setfilter(descr, &fp) == -1) {
		fprintf(stderr, "Couldn't install filter %s: %s\n",
		    filter_exp, pcap_geterr(descr));
		exit(EXIT_FAILURE);
	}

	/* Precarga de memoria*/
	
	//load_mem();
	int i = 0;
	//printf("prueba load mem\n");

	struct Flujo *item = (struct Flujo*) malloc(SIZE*sizeof(struct Flujo));

	for (i = 0; i < SIZE; i++){
		item->id_f = 0;
		hashArrayFlujo[i] = item;
		item ++;
	}	
		//printf("fin prueba load mem\n");

	retorno=pcap_loop(descr,NO_LIMIT,analizar_paquete,NULL);
	switch(retorno)	{
		case OK:
			//printf("Traza leída\n");
			break;
		case PACK_ERR: 
			//printf("Error leyendo paquetes\n");
			break;
		case BREAKLOOP: 
			//printf("pcap_breakloop llamado\n");
			break;
	}
	//printf("Se procesaron %"PRIu64" paquetes.\n\n", contador);
	pcap_close(descr);

	return OK;
}



// Analizador paquete
void analizar_paquete(u_char *user,const struct pcap_pkthdr *hdr, const uint8_t *pack)
{
	(void)user;
	//printf("*******************************************************\n");
	//printf("-------------------------------------------------------\n");
	//printf("Nuevo paquete capturado el %s\n", ctime((const time_t *) & (hdr->ts.tv_sec)));
	
	contador++;

	int flag_cabecera = 1; // si esta a 0 se añade la cabecera al txt si esta a 1 solo hay payload

	int hash_flujo = 0;
	int id_flujo = 0;

	// Variables para bucles
	int i = 0;
	int j = 0;

	// Variables para cond logicas
	int offset = 0;		// tag para ver si tiene paquete ip 
	int Udp = 0;
	int Tcp = 0;	// Da warning porque esta comentada la parte de analizar tcp

	// Variables para guardar longitud de paquete
	u_int ip_size = 0;
	u_int udp_size = 8;

	// Variables para guardar en tabla
	//char hostname[100] = {"NULL"};
	int key = 0;
	double timesec = 0;
	double timeus = 0;
	double time = 0;

	int ipproto = 0;

	int src_addr[4] = {0,0,0,0};
	int dst_addr[4] = {0,0,0,0};
	int src_port = 0;
	int dst_port = 0;
	//uint64_t cid = 0;
	//int version = 0;
	int ip_len = 0;
	int ip_header_len = 0;
	int udp_len = 0;
	int udp_header_len = 0;

	int size_tcp = 0;

	timesec = hdr->ts.tv_sec;
	timeus = hdr->ts.tv_usec;
	time = timesec + timeus*0.000001;

	/*Para campos ETH se usa casting ya que siempre siguen el mismo orden */
	const struct sniff_ethernet *ethernet;
	ethernet = (struct sniff_ethernet*)(pack);

	// Hay que comprobar que el paquete sea IP, si no se descarta.

	// if(ethernet->ether_type != 8){
	// 	//printf("No es un paquete IP, no se analiza");
	// 	//printf("\n");
	// 	return;
	// }
	// else{
		//printf("-------------------------------------------------------\n");
		//printf("Es un paquete IP, se analiza:\n");

	/*Para campos IP se usa casting ya que siempre siguen el mismo orden */
	const struct sniff_ip *ip;
	ip = (struct sniff_ip*)(pack + ETH_HLEN);
/*
	//printf("Version IP= ");
	//printf("%u", (((ip)->ip_vhl >> 4) & 0x0f));
	//printf("\n");
	//printf("IP Longitud de Cabecera= ");
*/		
	ip_size = 4*(ip->ip_vhl&0xf);
	ip_header_len = ip_size;
	////printf("ip header size%d\n",ip_size );
/*		//printf("%u Bytes", ip_size);
	//printf("\n");

	//printf("IP Longitud Total= ");
	//printf("%u Bytes", ntohs(ip->ip_len));
*/
	ip_len = ntohs(ip->ip_len);
	////printf("ip len: %d\n",ip_len );
/*
	//printf("\n");

	//printf("Posicion= ");
*/
	offset = 8*(ntohs((ip->ip_off))&0x1FFF);

	if(ip->ip_p == 17){			
		//printf("Es un paquete UDP\n");
		Udp = 1;
		ipproto = ip->ip_p;
	}
	else if(ip->ip_p == 6){		
		//printf("Es un paquete TCP\n");
		Tcp = 1;
		ipproto = ip->ip_p;
	}
	else{
		//printf("No es un paquete UDP ni TCP, no lo analizamos\n");
		Tcp = 0;
		Udp = 0;
		//return;
	}

	//printf("Direccion IP Origen= ");
	//printf("%u", ip->ip_src[0]);
	src_addr[0] = ip->ip_src[0];
	for (i = 1; i <IP_ALEN; i++) {
		//printf(".%u", ip->ip_src[i]);
		src_addr[i] = ip->ip_src[i];
	}
	//printf("\n");

	//printf("Direccion IP Destino= ");
	//printf("%u", ip->ip_dst[0]);
    dst_addr[0] = ip->ip_dst[0];
	for (j = 1; j <IP_ALEN; j++) {
		//printf(".%u", ip->ip_dst[j]);
    	dst_addr[j] = ip->ip_dst[j];
	}	
	//printf("\n");

	if(Udp == 1 && offset == 0){ 
		//printf("-------------------------------------------------------");
		//printf("\n");

		/*Para campos UDP se usa casting ya que siempre siguen el mismo orden */
		const struct sniff_udp *udp;
		udp = (struct sniff_udp*)(pack + ETH_HLEN + ip_size);

		//printf("Es un paquete UDP, se analiza:");
		//printf("\n");		
		//printf("Puerto Origen= ");		
		//printf("%u", ntohs(udp->udp_sport));
		src_port = ntohs(udp->udp_sport);
		//printf("\n");
	
		//printf("Puerto Destino= ");
		//printf("%u", ntohs(udp->udp_dport));
		dst_port = ntohs(udp->udp_dport);
		//printf("\n");
		
/*
		//printf("Longitud= ");
				
		//printf("%u", ntohs(udp->udp_length));
		//printf("\n");
*/			

		udp_len = ntohs(udp->udp_length);
		////printf("udp len: %d\n",udp_len );
		udp_header_len = udp_size;
		////printf("udp header size %d\n",udp_header_len );


		//printf("-------------------------------------------------------");
		//printf("\n");
	
	} // Cierre de cond udp

	else if(Tcp == 1 && offset == 0){
			
		//printf("-------------------------------------------------------");
		//printf("\n");

		const struct sniff_tcp *tcp;
		tcp = (struct sniff_tcp*)(pack + ETH_HLEN + ip_size);

		//printf("Es un paquete TCP, se analiza:");

		//printf("\n");
		//printf("Puerto Origen= ");

		//printf("%u", ntohs(tcp->th_sport));
		src_port = ntohs(tcp->th_sport);

		//printf("\n");

		//printf("Puerto Destino= ");

		//printf("%u", ntohs(tcp->th_dport));
		dst_port = ntohs(tcp->th_dport);
		//printf("\n");


		size_tcp = TH_OFF(tcp)*4;
		//printf("Longitud de la cabecera tcp: %d \n",size_tcp);
		
	}


	//printf("Quintupla del Flujo = %d %d.%d.%d.%d %d.%d.%d.%d %u %u \n",ipproto,src_addr[0],src_addr[1],src_addr[2],src_addr[3],dst_addr[0],dst_addr[1],dst_addr[2],dst_addr[3],src_port,dst_port);
	key =  src_addr[0]*10000+ src_addr[1]*10000;
	key +=  src_addr[2]*10000+ src_addr[3]*10000;
	key +=  dst_addr[0]+ dst_addr[1];
	key +=  dst_addr[2]+ dst_addr[3];

	key +=  src_port*10000 +  dst_port + ipproto;
	//printf("Key: %d\n",key);

	// Calcular el hash del flujo
	hash_flujo = key % SIZE;

	// Si flujo esta creado recupero su ID si no genero nuevo id

	if (hashArrayFlujo[hash_flujo]->id_f == 0){
		contador_flujos = contador_flujos + 1;
		hashArrayFlujo[hash_flujo]->id_f = contador_flujos;
		id_flujo = hashArrayFlujo[hash_flujo]->id_f;

		char dir[] = {0};
		sprintf(dir, "%s_%d",token, id_flujo);
		//printf("dir: %s\n", dir);

		int result = mkdir(dir,0777);
		//printf("Ha creado dir\n");
	}
	else{
		id_flujo = hashArrayFlujo[hash_flujo]->id_f;
	}

	//printf("id_flujo: %d\n",id_flujo);

	//printf("Guardo archivo: %s_%d_%ld\n",token,id_flujo,contador);

	char num = 0;

	// Para quitar las cabeceras ETH,IP,UDP/TCP 
	int ini = ETH_HLEN + ip_size;
	if (ip->ip_p==17){//UDP
		ini = ini + 8;
	}
	if (ip->ip_p==6){//TCP
		ini = ini + size_tcp;
	}

	// según flag cabcera se añade o no la cabcera al bin
	int suma_cabecera = 0;
	if(flag_cabecera == 0){

		suma_cabecera = 0;
	}
	else if(flag_cabecera == 1){

		suma_cabecera = ini;

	}


	char binario_inicial[1024] = {0};
	char senial_final[1024] = {0};

    for(int k=0; k<6;k++){
		char buffer[100]={0};

		sprintf(buffer, "%s_%d/%ld_A%d.bin",token, id_flujo, contador, k);
		//printf("%s\n", buffer);
		FILE *file = fopen(buffer,"w");


		int tam_img = 1024;



		if(k==0){
			int m=0;
			for (i = 0 + suma_cabecera; i < tam_img + suma_cabecera; i++){
				if(i < hdr->len){
					//printf("guardar pack[%d] = \n",i);
					// fprintf(file,"%c", pack[i]);
					binario_inicial[m] = pack[i];
					m++;
				}
				else{
					//printf("guardar %d = 0\n",i);
					// fprintf(file,"%c", num);
					binario_inicial[m] = num;
					m++;

				}
			}

			memcpy(senial_final, binario_inicial, sizeof(binario_inicial));



		}else if(k==1){
			// Horizontal flip
			// printf("\nHorizontal flip\n");
			// printf("Señal original:\n");
			// for (int i = 0; i < SIGNAL_LENGTH; i++) {
			//     printf("%c ", binario_inicial[i]);
			// }
			// printf("\n\n");
			// Intercambia los valores de izquierda a derecha
			char senial_flip[SIGNAL_LENGTH] = {0};
    		horizontal_flip(binario_inicial, SIGNAL_LENGTH, senial_flip);
			memcpy(senial_final, senial_flip, sizeof(senial_flip));
			// printf("Señal después del intercambio:\n");
    		// for (int i = 0; i < SIGNAL_LENGTH; i++) {
    		//     printf("%c ", binario_inicial[i]);
    		// }
    		// printf("\n");
			
			// curva_hilbert_relleno(binario_inicial, x, y,curva[DIM][DIM]);


		}else if(k==2){
			// Permutación
			// printf("\nPermutación\n");
			float alpha = 0.6; // Ejemplo de valor de alpha
			
			// printf("Señal original:\n");
    		// for (int i = 0; i < SIGNAL_LENGTH; i++) {
    		//     printf("%c ", binario_inicial[i]);
    		// }
    		// printf("\n\n");

    		// Realiza la permutación de la cadena
			char senial_permutada[SIGNAL_LENGTH] = {0};
    		permute_sequence(alpha, binario_inicial, SIGNAL_LENGTH, senial_permutada);
			memcpy(senial_final, senial_permutada, sizeof(senial_permutada));
    		// printf("Señal después de la permutación:\n");
    		// for (int i = 0; i < SIGNAL_LENGTH; i++) {
    		//     printf("%c ", binario_inicial[i]);
    		// }
    		// printf("\n");
			

		}else if(k==3){
			// Interpolacion
			// printf("\nInterpolación\n");

			// printf("Señal original:\n");
    		// for (int i = 0; i < SIGNAL_LENGTH; i++) {
    		//     printf("%c ", binario_inicial[i]);
    		// }
    		// printf("\n\n");
			// Interpola la señal
			char senial_interpolada[SIGNAL_LENGTH*2] = {0};
    		interpolate_signal(binario_inicial, SIGNAL_LENGTH, senial_interpolada);
			
			// printf("Señal después de la interpolación:\n");
    		
			// for (int i = 0; i < SIGNAL_LENGTH * 2 - 1; i++) {
    		//     printf("%c ", senial_interpolada[i]);
    		// }
    		// printf("\n\n");

			// Muestrea una nueva secuencia de longitud T=1024 y guarda la señal muestreada en otra cadena
    		// Inicializa la semilla del generador de números aleatorios
    		int numero;
			// srand(time(NULL));

	    	// Genera un número aleatorio en el rango de 0 a 2047
    		numero = rand() % 2048;

			// printf("%d\n", numero);
			int T = 1024; // Longitud de la nueva secuencia
    		char senial_muestreda[SIGNAL_LENGTH];
    		sample_sequence(senial_interpolada, numero, T, senial_muestreda);
			memcpy(senial_final, senial_muestreda, sizeof(senial_muestreda));
    		// printf("Señal muestreada:\n");
    		// for (int i = 0; i < T; i++) {
    		//     printf("%c ", senial_muestreda[i]);
    		// }
    		// printf("\n");
			// Dormir durante un segundo
    		// sleep(1);


		}else if(k==4){
			// Traslacion
			// printf("\nTraslación\n");

			int alpha = 5; // Ejemplo de valor de alpha

			// printf("Señal original:\n");
    		// for (int i = 0; i < SIGNAL_LENGTH; i++) {
    		//     printf("%c ", binario_inicial[i]);
    		// }
    		// printf("\n\n");

    		// Realiza la traslación del segmento
			// Copiar los valores de la cadena de origen a la cadena de destino
    		char senial_traslada[SIGNAL_LENGTH] = {0};
			memcpy(senial_traslada, binario_inicial, sizeof(binario_inicial));

			translate_segment(alpha, senial_traslada, SIGNAL_LENGTH);
			memcpy(senial_final, senial_traslada, sizeof(senial_traslada));
    		// printf("Señal después de la traslación:\n");
    		// for (int i = 0; i < SIGNAL_LENGTH; i++) {
    		//     printf("%c ", senial_traslada[i]);
    		// }
    		// printf("\n");
			// sleep(1);

		}

		 // Escribir la matriz en el archivo
    	for (int i = 0; i < tam_img; i++) {
        	fputc(senial_final[i], file);
    	}

		fclose(file);
	
	}

	

}


// funciones de la curva de hilbert
//Función para convertir d a coordenadas x e y
void d2xy(int n, int d, int *x, int *y) {
    int rx, ry, s, t = d;
    *x = *y = 0;
    for (s = 1; s < n; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

void rot(int n, int *x, int *y, int rx, int ry) {
    int t;
    if (ry == 0) {
        if (rx == 1) {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        t = *x;
        *x = *y;
        *y = t;
    }
}


// Función para realizar un horizontal flip a la señal
void horizontal_flip(char signal[], int length, char senial_flip[]) {
    for (int i = 0; i < length / 2; i++) {
        senial_flip[i] = signal[length - i - 1];
        senial_flip[length - i - 1] = signal[i];
    }
}

// Función para obtener el valor de N según la descripción dada
int get_Np(float alpha) {
    float ai[] = {0.15, 0.45, 0.75, 0.9};
    int N = 2;
    for (int i = 0; i < sizeof(ai) / sizeof(ai[0]); i++) {
        if (ai[i] <= alpha) {
            N = i + 2;
        }
    }
    return N;
}

// Función para realizar la permutación de la cadena
void permute_sequence(float alpha, char signal[], int length, char signal_permuted[]) {
    int N = get_Np(alpha); // Calcula N según el valor de alpha
    int n = 2 + rand() % N; // Genera un número aleatorio en el rango [2, N]

    // printf("Realizando permutación...\n");
    // printf("Número de segmentos (n): %d\n\n", n);

    // Calcula la longitud de cada segmento
    int segment_length = length / n;

    // Arreglo para almacenar los índices de inicio de los segmentos
    int segment_starts[n];

    // Genera índices aleatorios de inicio para los segmentos
    for (int i = 0; i < n; i++) {
        segment_starts[i] = rand() % length;
    }

    // Ordena los índices de inicio de los segmentos de manera ascendente
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (segment_starts[j] > segment_starts[j + 1]) {
                int temp = segment_starts[j];
                segment_starts[j] = segment_starts[j + 1];
                segment_starts[j + 1] = temp;
            }
        }
    }

    // Componer la nueva muestra x' concatenando los segmentos
    char permuted_signal[length];
    int index = 0;
    for (int i = 0; i < n; i++) {
        for (int j = segment_starts[i]; j < segment_starts[i] + segment_length && j < length; j++) {
            permuted_signal[index++] = signal[j];
        }
    }

    // Copia el resultado de la permutación de vuelta a la señal original
    for (int i = 0; i < length; i++) {
        signal_permuted[i] = permuted_signal[i];
    }
}

//  Función para realizar la interpolación en una señal representada como una cadena de caracteres
void interpolate_signal(char signal[], int length, char interpolated_signal[]) {

    for (int i = 0; i < length ; i++) {
        interpolated_signal[i * 2] = signal[i];
        interpolated_signal[i * 2 + 1] = (signal[i] + signal[i + 1]) / 2;
    }

}

// Función para muestrear una nueva secuencia de longitud T y guardarla en otra cadena
void sample_sequence(char signal[], int n, int T, char sampled_signal[]) {
	for (int i = 0; i < T; i++) {
        sampled_signal[i] = signal[i];
    }
}


// Función para obtener el valor de N según la descripción dada
int get_Nt(int alpha) {
    int ai[] = {0, 15, 0, 3, 0, 5, 0, 8};
    int N = 1;
    for (int i = 0; i < sizeof(ai) / sizeof(ai[0]); i++) {
        if (ai[i] <= alpha) {
            N = i + 1;
        }
    }
    return N;
}

// Función para realizar la traslación del segmento
void translate_segment(int alpha, char signal[], int length) {
    int N = get_Nt(alpha); // Calcula N según el valor de alpha
    int n = 1 + rand() % N; // Genera un número aleatorio en el rango [1, N]
    char b = rand() % 2 == 0 ? 'l' : 'r'; // Genera una dirección aleatoria (l: izquierda, r: derecha)
    int t = 0; // Genera un punto de partida aleatorio en el rango [0, T]

    // printf("Realizando traslación...\n");
    // printf("Número de pasos (n): %d\n", n);
    // printf("Dirección (b): %c\n", b);
    // printf("Punto de partida (t): %d\n\n", t);

	char senial_aux[SIGNAL_LENGTH] = {0};
	// memcpy(senial_aux, signal, sizeof(signal));
    for (int i = 0; i < SIGNAL_LENGTH; i++) {
		senial_aux[i] = signal[i]; 
    }

    if (b == 'l') { // Si la dirección es izquierda
        for (int i = t-n, j=0; i < SIGNAL_LENGTH; i++, j++) {
			if(i+n>=SIGNAL_LENGTH){
				signal[i] = '\0';
			}else if(i<=0){
				signal[0] = senial_aux[t+j];

			}else{
				signal[i] = senial_aux[t+j];

			}

        }
    } else { // Si la dirección es derecha
        char x = senial_aux[t]; // Guarda el valor en la posición t
		for (int i = t, j=0; i<SIGNAL_LENGTH; i++, j++) {
			if(i<(t+n)){
				signal[i] = x;
				j=0;
			}else{
				signal[i] = senial_aux[t+j];

			}
			
        }
    }
}


// Función para agregar ruido gaussiano
void add_gaussian_noise(char cadena[], int caracteristica, float alpha, float sigma) {
    srand(time(NULL));
    for (int i = 0; i < SIGNAL_LENGTH; i++) {
        if (rand() % 2 == 0) { // Muestrear aleatoriamente una característica
            float ruido = sigma * sqrt(alpha) * ((float)rand() / RAND_MAX); // Generar ruido gaussiano
            // cadena[i] += ruido; // Agregar ruido a Tamaño
			if (caracteristica == 1) { // Si la característica es Tamaño
                cadena[i] += ruido; // Agregar ruido a Tamaño
            }
            // Puedes agregar más condiciones para otras características si es necesario
        }
    }
}