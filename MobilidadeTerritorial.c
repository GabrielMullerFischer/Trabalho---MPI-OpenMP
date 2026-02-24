#include <mpi.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>

// Nome do integrantes do grupo: Gabriel Muller Fischer, Pedro Ivo Kuhn

// --- DEFINIÇÕES E ESTRUTURAS ---
struct Cell {
    int tipo;          // 0: aldeia, 1: pesca, etc.
    double recurso;    // Valor do recurso
};

struct Agent {
    int id;
    int x, y;          // Posição global ou local
    double energia;
};

// Parâmetros da Simulação
const int W = 1000;    // Largura Global
const int H = 1000;    // Altura Global
const int T = 100;     // Total de Ciclos
const int S = 20;      // Tamanho da Estação

// Função de Carga Sintética (Item 2.4 do PDF)
// Executa laço inútil proporcional ao recurso para gastar CPU
void executar_carga_sintetica(double recurso) {
    long long iteracoes = (long long)(recurso * 1000.0); 
    double lixo = 0.0;
    for(long long i = 0; i < iteracoes; i++) {
        lixo += sin(i) * cos(i); // Cálculo matemático pesado
    }
    // O compilador não pode otimizar e remover isso
    if (lixo > 100000) asm(""); 
}

int main(int argc, char** argv) {
    // 1. Inicializar MPI [cite: 79-81]
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 2. Particionar o Grid (Divisão por Linhas para simplificar) [cite: 89, 95]
    // Cada processo fica com uma faixa de linhas (H / size)
    int local_H = H / size; 
    int start_y = rank * local_H;
    int end_y = start_y + local_H;

    // Alocar Grid Local (com halo/bordas extras para comunicação)
    // +2 linhas para as bordas (uma em cima, uma embaixo)
    std::vector<std::vector<Cell>> local_grid(local_H + 2, std::vector<Cell>(W));

    // 3. Inicializar Grid Local [cite: 97]
    for(int i = 1; i <= local_H; i++) {
        for(int j = 0; j < W; j++) {
            // Lógica de inicialização baseada na posição global
            local_grid[i][j].recurso = (double)(rand() % 100); 
            local_grid[i][j].tipo = 0; 
        }
    }

    // 4. Inicializar Agentes Locais [cite: 109]
    std::vector<Agent> meus_agentes;
    // (Adicione lógica para criar agentes apenas se a posição inicial cair neste rank)

    int estacao = 0; // 0: Seca, 1: Cheia

    // 5. Loop Principal de Tempo [cite: 116]
    for (int t = 0; t < T; t++) {
        
        // 5.1 Atualizar Estação [cite: 125]
        if (t % S == 0) {
            if (rank == 0) estacao = !estacao;
            MPI_Bcast(&estacao, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

        // 5.2 Troca de Halo (MPI) 
        // Enviar linha 1 para o vizinho de cima, receber na linha 0
        // Enviar linha local_H para vizinho de baixo, receber na linha local_H+1
        // (Use MPI_Sendrecv aqui)

        // Buffers para migração de agentes
        std::vector<Agent> agentes_para_enviar_cima;
        std::vector<Agent> agentes_para_enviar_baixo;
        std::vector<Agent> agentes_que_ficaram;

        // 5.3 Processar Agentes (OpenMP) 
        // O pragma parallel cria as threads
        #pragma omp parallel 
        {
            // Buffers privados por thread para evitar condição de corrida
            std::vector<Agent> local_stay, local_send_up, local_send_down;

            #pragma omp for nowait
            for (size_t i = 0; i < meus_agentes.size(); i++) {
                Agent a = meus_agentes[i];
                
                // Pegar recurso da célula atual
                // Note: o índice Y local do agente deve ser ajustado para o grid local
                double rec = local_grid[a.y - start_y + 1][a.x].recurso;

                // Executar Carga Sintética [cite: 46]
                executar_carga_sintetica(rec);

                // Decidir movimento (simplificado)
                int dx = (rand() % 3) - 1; 
                int dy = (rand() % 3) - 1;
                a.x += dx; 
                a.y += dy;

                // Verificar se mudou de processo (MPI)
                if (a.y < start_y) {
                    local_send_up.push_back(a); // Migra para cima
                } else if (a.y >= end_y) {
                    local_send_down.push_back(a); // Migra para baixo
                } else {
                    local_stay.push_back(a); // Fica aqui
                }
            } // Fim do for paralelo

            // Região crítica para juntar os dados das threads nos vetores principais
            #pragma omp critical 
            {
                agentes_que_ficaram.insert(agentes_que_ficaram.end(), local_stay.begin(), local_stay.end());
                agentes_para_enviar_cima.insert(agentes_para_enviar_cima.end(), local_send_up.begin(), local_send_down.end());
                // ... fazer o mesmo para baixo
            }
        } // Fim da região paralela

        // Atualizar lista principal
        meus_agentes = agentes_que_ficaram;

        // 5.4 Migrar Agentes (MPI) [cite: 64]
        // Usar MPI_Send / MPI_Recv para enviar os vetores 'agentes_para_enviar...' 
        // para os vizinhos e receber novos agentes.
        
        // 5.5 Atualizar Grid Local (OpenMP) 
        #pragma omp parallel for collapse(2)
        for(int i = 1; i <= local_H; i++) {
            for(int j = 0; j < W; j++) {
                // Regenerar recurso baseado na estação
                local_grid[i][j].recurso += (estacao == 0 ? 0.1 : 0.5); 
            }
        }
        
        // Sincronização opcional para imprimir passo
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0) std::cout << "Ciclo " << t << " concluido." << std::endl;
    }

    // 6. Finalizar MPI [cite: 81]
    MPI_Finalize();
    return 0;
}