#include <mpi.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <ctime>

Nome dos participantes: Gabriel Muller Fischer, Pedro Ivo Kuhn

// --- DEFINIÇÕES E CONSTANTES ---
const int TIPO_ALDEIA = 0;
const int TIPO_PESCA = 1;
const int TIPO_COLETA = 2;
const int TIPO_ROCADO = 3;
const int TIPO_INTERDITADO = 4;
const int TIPO_VARZEA = 5;

const int W = 1000;
const int H = 1000;
const int T = 100;
const int S = 20;
const long long MAXIMO_INTERACOES = 200000; 

struct Celula { // pedaço do terreno (Recurso e tipo)
    int tipo;
    double recurso;
};

struct Agente { // O indíduo
    int id;
    int x, y;
    double energia;
};

void executar_carga_sintetica(double recurso) {
    long long iteracoes = (long long)(recurso * 1000.0); 
    if (iteracoes > MAXIMO_INTERACOES) iteracoes = MAXIMO_INTERACOES;
    double lixo = 0.0;
    for(long long i = 0; i < iteracoes; i++) lixo += sin(i) * cos(i); 
    if (lixo > 10000000) asm(""); 
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // inicia o ambiente distribuido
    double t_inicio = MPI_Wtime(); // Salva o tempo atual em segundos para no final saber o tempo de simulação

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Pergunta "Quem sou eu?"
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Pergunta "Quantos somos?"

    // Tipos MPI
    MPI_Datatype MPI_TIPO_AGENTE;
    MPI_Type_contiguous(sizeof(Agente), MPI_BYTE, &MPI_TIPO_AGENTE); // Aqui mostra o que é um Agente para o MPI
    MPI_Type_commit(&MPI_TIPO_AGENTE); // salva a definição de agente

    MPI_Datatype MPI_TIPO_CELULA;
    MPI_Type_contiguous(sizeof(Celula), MPI_BYTE, &MPI_TIPO_CELULA);
    MPI_Type_commit(&MPI_TIPO_CELULA);

    // Particionamento
    int local_H = H / size; // calcula quantas linhas cada processo vai cuidar
    int inicio_y = rank * local_H; // onde que começa
    int fim_y = inicio_y + local_H; // onde que termina

    // Matriz local, com +2 de borda fantasma
    // linha 0 e local_H+1 armazenam cópias dos dados dos vizinhos, assim 'vendo' fora do processo atual
    std::vector<std::vector<Celula>> grid_local(local_H + 2, std::vector<Celula>(W));

    // 1. Inicializar Grid
    // preenchendo apenas as celulas reais, as bordas 0 e 251 ficam vazias até ter trocas de msg
    srand(rank * 100 + 1); // semente aleatoria usando o rank do processo para gerar um terreno diferente entre os processos
    for(int i = 1; i <= local_H; i++) {
        for(int j = 0; j < W; j++) {
            grid_local[i][j].recurso = (double)(rand() % 100); 
            grid_local[i][j].tipo = rand() % 6; 
        }
    }

    // 2. Inicializar Agentes
    std::vector<Agente> meus_agentes;
    int total_agentes_simulacao = 10000;
    srand(12345); // Mesma semente para todos terem os mesmos numeros aleatorios

    for (int k = 0; k < total_agentes_simulacao; k++) {
        int x_global = rand() % W;
        int y_global = rand() % H;
        // Essa coordenada cai no meu pedaço de terra?
        // Sim: Cria o agente e guarda na memória. Não: ignora e passa pro próximo
        if (y_global >= inicio_y && y_global < fim_y) {
            Agente agente;
            agente.id = k;
            agente.x = x_global;
            agente.y = y_global;
            agente.energia = 100.0; 
            meus_agentes.push_back(agente);
        }
    }
    
    // Após gerar tudo muda a semente para o relógio + rank
    // Garantindo que o comportamento dos agentes seja imprevisível (vivo)
    srand(time(NULL) + rank);
    int estacao = 0; 

    // --- SETUP INICIAL ---
    if (rank == 0) {
        int threads = 1;
        #pragma omp parallel
        { 
            #pragma omp master 
            threads = omp_get_num_threads(); // apenas a thread principal conta quantas threads realmente alocou
        }
        std::cout << "=== SIMULACAO INICIADA ===" << std::endl;
        std::cout << "Processos MPI: " << size << std::endl;
        std::cout << "Threads/No: " << threads << std::endl;
        std::cout << "População Inicial: " << total_agentes_simulacao << std::endl;
        std::cout << "==========================" << std::endl;
    }

    long long total_perdidos_local = 0; 

    // ================= LOOP PRINCIPAL =================
    for (int t = 0; t < T; t++) { // cada t é um dia na vida dos agentes

        // A. Atualizar Estação
        if (t % S == 0) { // Verifica se é hora de mudar de estação (a cada S dias)
            if (rank == 0) estacao = !estacao; // Apenas o mestre troca a estação
            MPI_Bcast(&estacao, 1, MPI_INT, 0, MPI_COMM_WORLD); // Avisa os outros processos que a estação mudou
        }

        // B. Troca de Halo (Dummy para simular custo de rede)
        int rank_superior = (rank == 0) ? MPI_PROC_NULL : rank - 1; // MPI_PROC_NULL significa ninguem, se eu sou o rank 0 não tenho ninguem acima
        int rank_inferior = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

        // Estou mandando minha borda superior para o vizinho de cima, e esperando 
        // o vizinho de baixo me mandar a borda superior dele (que será minha nova borda inferior).
        MPI_Sendrecv(grid_local[1].data(), // O que vai ser enviado (buffer de envio)
                                        W, // Quantos itens
                          MPI_TIPO_CELULA, // Qual o tipo de dado
                            rank_superior, // Para quem (destino)
                                        0, // Etiqueta do envio
           grid_local[local_H + 1].data(), // Onde guardo o que chegar (bufeer de recebimento)
                                        W, // Quantos itens espero receber
                          MPI_TIPO_CELULA, // Qual o tipo esperado
                            rank_inferior, // De quem espero receber
                                        0, // Etiqueta de recebimento
                           MPI_COMM_WORLD, // Comunicador (O universo dos processos)
                        MPI_STATUS_IGNORE);// Status
        // Mandando minha borda inferior para o vizinho de baixo e esperando o de cima para para preencher a minha inferior
        MPI_Sendrecv(grid_local[local_H].data(), W, MPI_TIPO_CELULA, rank_inferior, 1, 
                     grid_local[0].data(),       W, MPI_TIPO_CELULA, rank_superior, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Se o código estivesse 100% correto conforme o trabalho pede, em vez de d_snd (um double único), ele estaria enviando a linha inteira da matriz (grid_local[1] ou grid_local[local_H]) .

        // movimentação dos agentes (são os buffers para a troca de agentes)
        std::vector<Agente> agentes_para_enviar_cima; // agentes que ficaram com a coordenada y < inicio_y
        std::vector<Agente> agentes_para_enviar_baixo; // agentes qie ficatam com coordenada y >= fim_y
        std::vector<Agente> agentes_que_ficaram; // agentes que ficaram dentro dos limites

        // C. Processamento de Agentes
        #pragma omp parallel 
        {
            // Vetores para cada thread
            std::vector<Agente> agentes_que_ficam_local, agentes_para_cima_local, agentes_para_baixo_local;
            
            // Pega o total de agentes e divide entre as threads
            #pragma omp for nowait // nowait fala para a thread continuar mesmo que as outras não tenham terminado
            for (size_t i = 0; i < meus_agentes.size(); i++) {
                Agente agente = meus_agentes[i];
                
                // SEED unica para cada thread
                unsigned int seed = 12345 + omp_get_thread_num() + t + rank; 

                // 1. Metabolismo (Gasta 5.0 por ciclo)
                agente.energia -= 5.0; 

                // agente sabe a posição global, com isso temos que calcular a posição local+borda
                int posicao_local = agente.y - inicio_y + 1; // a linha da matriz
                if(posicao_local < 1) posicao_local = 1; 
                if(posicao_local > local_H) posicao_local = local_H;

                double comida_obtida = 0.0;
                double fome = 100.0 - agente.energia; 
                double apetite = (fome > 15.0) ? 15.0 : fome; // Come até 15.0

                // 2. Consumo Atômico
                // Trecho que apenas uma thread pode entrar
                #pragma omp critical(consumo)
                {
                    double disponivel = grid_local[posicao_local][agente.x].recurso;
                    double vai_comer = (disponivel >= apetite) ? apetite : disponivel;
                    grid_local[posicao_local][agente.x].recurso -= vai_comer;
                    comida_obtida = vai_comer;
                }
                
                agente.energia += comida_obtida;

                // Garante que a energia nunca passe de 100%
                if (agente.energia > 100.0) {
                    agente.energia = 100.0;
                }

                // simula o trabalho para pegar a comida...
                executar_carga_sintetica(comida_obtida * 5.0 + 10.0);

                // 3. Checagem de Morte
                // o continue, faz ir para o proximo 'i', e com isso o agente não é adicionado em quem ficou/foi
                if (agente.energia <= 0.0) continue; 

                // 4. Inteligência de Movimento
                bool mover = true;
                // Se o lugar é bom (>5 comidas), 80% chance de ficar
                if (comida_obtida > 5.0) { 
                    if ((rand_r(&seed) % 100) < 80) mover = false; 
                }

                if (mover) {
                    // rand gera 0, 1, 2. Subtraindo 1. Ele pode ir para todas as 8 células vizinhas
                    int movimento_x = (rand_r(&seed) % 3) - 1; 
                    int movimento_y = (rand_r(&seed) % 3) - 1;
                    
                    int candidato_x = agente.x; 
                    int candidato_y = agente.y;

                    // --- BORDAS GLOBAIS ---
                    if (agente.x + movimento_x >= 0 && agente.x + movimento_x < W) candidato_x += movimento_x;
                    if (agente.y + movimento_y >= 0 && agente.y + movimento_y < H) candidato_y += movimento_y;

                    // --- AVALIAÇÃO DE VIZINHO ---
                    bool movimento_aprovado = true;
                    int destino_local = candidato_y - inicio_y + 1; // converte Y global para Y local
                    
                    // Só avalia se o vizinho estiver no grid_local + borda
                    if (destino_local >= 0 && destino_local <= local_H +1) {
                        
                        // A. Regras Físicas
                        int tipo_dest = grid_local[destino_local][candidato_x].tipo;
                        if (tipo_dest == TIPO_INTERDITADO) movimento_aprovado = false;
                        if (estacao == 1 && tipo_dest == TIPO_VARZEA) movimento_aprovado = false;

                        // B. Regras de Inteligência (Evita lugares pobres)
                        if (movimento_aprovado) {
                            double recurso_vizinho = grid_local[destino_local][candidato_x].recurso;
                            double recurso_atual = grid_local[posicao_local][agente.x].recurso;
                            // Se vizinho for muito pior, 90% chance de desistir
                            if (recurso_vizinho < (recurso_atual * 0.2)) {
                                if ((rand_r(&seed) % 100) < 90) movimento_aprovado = false;
                            }
                        }
                    }

                    if (movimento_aprovado) { 
                        agente.x = candidato_x; 
                        agente.y = candidato_y; 
                    }
                }

                // 5. Classificação MPI
                if (agente.y < inicio_y) agentes_para_cima_local.push_back(agente);  // se menor que a posição de cima, sobe
                else if (agente.y >= fim_y) agentes_para_baixo_local.push_back(agente); // se maior que a maior posição, desce
                else agentes_que_ficam_local.push_back(agente); // se não fica
            }

            // Junta as listas
            #pragma omp critical 
            {
                agentes_que_ficaram.insert(agentes_que_ficaram.end(), agentes_que_ficam_local.begin(), agentes_que_ficam_local.end());
                agentes_para_enviar_cima.insert(agentes_para_enviar_cima.end(), agentes_para_cima_local.begin(), agentes_para_cima_local.end());
                agentes_para_enviar_baixo.insert(agentes_para_enviar_baixo.end(), agentes_para_baixo_local.begin(), agentes_para_baixo_local.end());
            }
        }
        
        // Atualiza a lista dos agentes que não migraram
        meus_agentes = agentes_que_ficaram;

        // D. Perdas e Migração, se alguem saiu do mapa
        // Com as bordas globais ativadas, isso deve ser sempre 0
        long long perdidos_ciclo_local = 0;
        if (rank == 0) perdidos_ciclo_local += agentes_para_enviar_cima.size();
        if (rank == size - 1) perdidos_ciclo_local += agentes_para_enviar_baixo.size();
        
        if (perdidos_ciclo_local > 0) {
            #pragma omp atomic
            total_perdidos_local += perdidos_ciclo_local;
        }

        int n_sobe = agentes_para_enviar_cima.size();
        int n_desce = agentes_para_enviar_baixo.size();
        int r_sobe = 0, r_desce = 0;

        MPI_Sendrecv(&n_sobe, 1, MPI_INT, rank_superior, 10, &r_desce, 1, MPI_INT, rank_inferior, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&n_desce, 1, MPI_INT, rank_inferior, 20, &r_sobe, 1, MPI_INT, rank_superior, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<Agente> inc_sobe(r_sobe), inc_desce(r_desce);
        MPI_Sendrecv(agentes_para_enviar_cima.data(), n_sobe, MPI_TIPO_AGENTE, rank_superior, 11, inc_desce.data(), r_desce, MPI_TIPO_AGENTE, rank_inferior, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(agentes_para_enviar_baixo.data(), n_desce, MPI_TIPO_AGENTE, rank_inferior, 21, inc_sobe.data(), r_sobe, MPI_TIPO_AGENTE, rank_superior, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        meus_agentes.insert(meus_agentes.end(), inc_sobe.begin(), inc_sobe.end());
        meus_agentes.insert(meus_agentes.end(), inc_desce.begin(), inc_desce.end());

        // F. Atualização do Grid
        #pragma omp parallel for collapse(2) // collapse funde os dois for
        for(int i = 1; i <= local_H; i++) {
            for(int j = 0; j < W; j++) {
                int tipo = grid_local[i][j].tipo;
                double fator = 1.0; double bonus = 0.0;

                // --- ESTAÇÃO SECA ---
                if (estacao == 0) { 
                    if (tipo == TIPO_PESCA){
                        fator = 0.85; // Rio: Perde 15% por ciclo
                    }
                    else if (tipo == TIPO_VARZEA){
                        fator = 0.90; // Várzea: Perde 10% por ciclo
                    }
                    else if (tipo == TIPO_ROCADO){
                        fator = 0.95; // Roçado: Perde 5% por ciclo
                    }
                    else {
                        fator = 0.98; // Floresta: Perde 2% por ciclo
                    }
                    // Seca: Reduz proporcionalmente
                    grid_local[i][j].recurso *= fator;
                }
                // --- ESTAÇÃO CHEIA ---
                else {
                    if (tipo == TIPO_PESCA){
                        bonus = 15.0; // Rio: ganha +15 de recurso por ciclo
                    }
                    else if (tipo == TIPO_VARZEA){
                        bonus = 10.0; // Várzea: ganha +10 de recurso por ciclo
                    }
                    else if (tipo == TIPO_ROCADO){
                        bonus = 5.0;  // Roçado: ganha +5 de recurso por ciclo
                    }
                    else {
                        bonus = 2.0;  // Floresta: ganha +2 de recurso por ciclo
                    }
                    // Cheia: Aumenta proporcionalmente
                    grid_local[i][j].recurso += bonus;
                }

                // O recurso nunca passa de 100 (cheio) e nem fica negativo (vazio)
                if (grid_local[i][j].recurso > 100.0) grid_local[i][j].recurso = 100.0;
                if (grid_local[i][j].recurso < 0.0) grid_local[i][j].recurso = 0.0;
            }
        }

        // --- RELATÓRIO DO CICLO (Estação + Recursos) ---
        double ciclo_recurso_local = 0.0;
        #pragma omp parallel for reduction(+:ciclo_recurso_local) // cria uma variavel local para somar na orginal no final
        for(int i=1; i<=local_H; i++) 
            for(int j=0; j<W; j++) ciclo_recurso_local += grid_local[i][j].recurso;

        // soma os recursos de todos os computadores
        double ciclo_recurso_global = 0.0;
        MPI_Reduce(&ciclo_recurso_local, &ciclo_recurso_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "-> Ciclo " << t 
                      << " | Estacao: " << (estacao ? "CHEIA" : "SECA ") 
                      << " | Recurso Total: " << (long)ciclo_recurso_global 
                      << std::endl;
        }

        // Faz esperar até que todos cheguem aqui
        MPI_Barrier(MPI_COMM_WORLD);

    } // Fim do Loop

    // ================= RELATÓRIO FINAL DETALHADO =================
    long local_agentes = meus_agentes.size();
    long global_agentes = 0;
    double local_recurso = 0;
    double global_recurso = 0;

    // soma dos recursos local
    #pragma omp parallel for reduction(+:local_recurso)
    for(int i=1; i<=local_H; i++) 
        for(int j=0; j<W; j++) local_recurso += grid_local[i][j].recurso;

    // Armazena no rank0, a soma de todos os recrusos e agentes
    MPI_Reduce(&local_agentes, &global_agentes, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_recurso, &global_recurso, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // tempo final da simulação
    double t_fim = MPI_Wtime();

    if (rank == 0) {
        long mortos_fome = 10000 - (global_agentes);

        std::cout << "\n=== RELATORIO DE SIMULACAO ===" << std::endl;
        std::cout << "Processos MPI: " << size << std::endl;
        std::cout << "Ciclos simulados: " << T << std::endl;
        std::cout << "Tempo Total: " << (t_fim - t_inicio) << " segundos" << std::endl;
        std::cout << "-----------------------------" << std::endl;
        std::cout << "Agentes Vivos: " << global_agentes << std::endl;
        std::cout << "Agentes Mortos (Fome): " << mortos_fome << std::endl;
        std::cout << "TOTAL (Vivos + Mortos): " << 10000 << std::endl;
        std::cout << "-----------------------------" << std::endl;
        std::cout << "Total Recurso Restante: " << (long)global_recurso << std::endl;
        std::cout << "=============================" << std::endl;
    }

    MPI_Type_free(&MPI_TIPO_AGENTE);
    MPI_Type_free(&MPI_TIPO_CELULA);
    MPI_Finalize();
    return 0;
}