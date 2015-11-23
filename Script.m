echo on
clear

nNosEscondidos= [1 5 10 50];% Numero de nodos escondidos do experimento
taxasApr =  [0.01 0.1 0.5 0.9]; %Taxas de aprendizado do experimento

%fileID = fopen('configEtapa1.txt','w');

i=3;
j=1;


%for i=1:4
 %   for j=1:4
        %    Informacoes sobre a rede e os dados
        numEntradas   = 6;     % Numero de nodos de entrada
        numEscondidos = nNosEscondidos(i);     % Numero de nodos escondidos
        numSaidas     = 1;     % Numero de nodos de saida
        numTr         = 10922;   % Numero de padroes de treinamento
        numVal        = 5460;    % Numero de padroes de validacao
        numTeste      = 2797;    % Numero de padroes de teste
        
        echo off
        
        %    Lendo arquivos e armazenando dados em matrizes
        dadosTreinamento    = csvread('mammography-consolidated-training-oversample.csv')';    % Lendo arquivo de treinamento
        entradasTreinamento = dadosTreinamento(1:numEntradas, 1:numTr);
        saidasTreinamento   = dadosTreinamento((numEntradas + 1):(numEntradas + numSaidas), 1:numTr);
        
        dadosValidacao      = csvread('mammography-consolidated-validation-oversample.csv')';    % Mesmo processo para validacao
        entradasValidacao   = dadosValidacao(1:numEntradas, 1:numVal);
        saidasValidacao     = dadosValidacao((numEntradas + 1):(numEntradas + numSaidas), 1:numVal);
        
        dadosTeste          = csvread('mammography-consolidated-test-oversample.csv')';      % Mesmo processo para teste
        entradasTeste       = dadosTeste(1:numEntradas, 1:numTeste);
        saidasTeste         = dadosTeste((numEntradas + 1):(numEntradas + numSaidas), 1:numTeste);
        
        
        %   Criando a rede (para ajuda, digite 'help newff')
        
        for entrada = 1 : numEntradas;  % Cria 'matrizFaixa', que possui 'numEntradas' linhas, cada uma sendo igual a [0 1].
            matrizFaixa(entrada,:) = [0 1];
        end
        
        rede = newff(matrizFaixa,[numEscondidos numSaidas],{'logsig','logsig'},'traingdm','learngdm','mse');
        % matrizFaixa                    : indica que todas as entradas possuem valores na faixa entre 0 e 1
        % [numEscondidos numSaidas]      : indica a quantidade de nodos escondidos e de saida da rede
        % {'logsig','logsig'}            : indica que os nodos das camadas escondida e de saida terao funcao de ativacao sigmoide logistica
        % 'traingdm','learngdm'          : indica que o treinamento vai ser feito com gradiente descendente (backpropagation)
        % 'sse'                          : indica que o erro a ser utilizado vai ser SSE (soma dos erros quadraticos)
        
        % Inicializa os pesos da rede criada (para ajuda, digite 'help init')
        rede = init(rede);
        echo on
        %   Parametros do treinamento (para ajuda, digite 'help traingd')
        rede.trainParam.epochs   = 30000;    % Maximo numero de iteracoes
        rede.trainParam.lr       = taxasApr(j);  % Taxa de aprendizado
        rede.trainParam.goal     = 0;      % Criterio de minimo erro de treinamento
        rede.trainParam.max_fail = 10;      % Criterio de quantidade maxima de falhas na validacao
        rede.trainParam.min_grad = 0;      % Criterio de gradiente minimo
        rede.trainParam.show     = 10;     % Iteracoes entre exibicoes na tela (preenchendo com 'NaN', nao exibe na tela)
        rede.trainParam.time     = inf;    % Tempo maximo (em segundos) para o treinamento
        echo off
        fprintf('\nTreinando ...\n')
        
        conjuntoValidacao.P = entradasValidacao; % Entradas da validacao
        conjuntoValidacao.T = saidasValidacao;   % Saidas desejadas da validacao
        
        %   Treinando a rede
        [redeNova,desempenho,saidasRede,erros] = train(rede,entradasTreinamento,saidasTreinamento,[],[],conjuntoValidacao);
        % redeNova   : rede apos treinamento
        % desempenho : apresenta os seguintes resultados
        %              desempenho.perf  - vetor com os erros de treinamento de todas as iteracoes (neste exemplo, escolheu-se erro SSE)
        %              desempenho.vperf - vetor com os erros de validacao de todas as iteracoes (idem)
        %              desempenho.epoch - vetor com as iteracoes efetuadas
        % saidasRede : matriz contendo as saidas da rede para cada padrao de treinamento
        % erros      : matriz contendo os erros para cada padrao de treinamento
        %             (para cada padrao: erro = saida desejada - saida da rede)
        % Obs.       : Os dois argumentos de 'train' preenchidos com [] apenas sao utilizados quando se usam delays
        %             (para ajuda, digitar 'help train')
        
        fprintf('\nTestando ...\n');
        
        %    Testando a rede
        [saidasRedeTeste,Pf,Af,errosTeste,desempenhoTeste] = privatesim(redeNova,entradasTeste,[],[],saidasTeste);
        % saidasRedeTeste : matriz contendo as saidas da rede para cada padrao de teste
        % Pf,Af           : matrizes nao usadas neste exemplo (apenas quando se usam delays)
        % errosTeste      : matriz contendo os erros para cada padrao de teste
        %                  (para cada padrao: erro = saida desejada - saida da rede)
        % desempenhoTeste : erro de teste (neste exemplo, escolheu-se erro SSE)
        
        fprintf('MSE para o conjunto de treinamento: %6.5f \n',desempenho.perf(length(desempenho.perf)));
        fprintf('MSE para o conjunto de validacao: %6.5f \n',desempenho.vperf(length(desempenho.vperf)));
        fprintf('MSE para o conjunto de teste: %6.5f \n',desempenhoTeste);
        
       % fprintf(fileID,'%f %f %f\n', nNosEscondidos(i),taxasApr(j), desempenhoTeste);
        
        
         plotroc(saidasTeste, saidasRedeTeste)
         [c,cm,ind,per] = confusion(saidasTeste, saidasRedeTeste);
         disp('Confusion Matrix')
         figure, plotperform(desempenho)
        disp(cm)
 %   end
%end

%fclose(fileID);