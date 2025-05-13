% Here we solve the cable equation numerically.

close all; clear all;

startTime = now; % фиксируем момент старта
disp(['Начало расчета: ', datestr(startTime, 'dd-mm-yyyy HH:MM:SS')]);

tic; % Начало измерения времени

rng(45); % Сид для воспроизводимости

startFromCheckpoint = false;
%startFromCheckpoint = true;
checkpointFileName = 'Checkpoint.mat';
checkpointStep = 10.0; % save every ... ms

if (startFromCheckpoint)
    if isfile(checkpointFileName)
        disp('Checkpoint loading...');
        load(checkpointFileName);
        dt = 5e-5; % in ms
        disp(['Continue from t=', num2str(t*dt), ' ms']);
        
        numRegular = sum(strcmp(spikePatterns, 'Regular'));
        numIrregular = sum(strcmp(spikePatterns, 'Irregular'));
        numSemiregular = sum(strcmp(spikePatterns, 'Semiregular'));
    else
        disp(['No checkpoint file detected: ', checkpointFileName]);
    end
else
    disp('Starting without checkpoint...');

    dx = 10e-2; % mm
    X = 300; % mm
    %X = 1000; % mm
    l = 2e-3; % node length in mm (длина узла Ранвье)
    x = dx.*[1:ceil(X/dx)];

    maxTime = 300.0; % ms
    %maxTime = 5.0;
    %maxTime = 1000.0;
    N = 100; % Количество аксонов
    %active_axons = 1:10; % Список активных аксонов
    active_axons = 1:100;
    stimDelayInterval = [0.0, 0.0]; % Задержки стимула в мс
    meanISI = 15; % Средний ISI (мс), используемый для паттернов

    V = zeros(N,numel(x));
    I = zeros(N,numel(x));

    dt = 5e-5; % in ms

    t = 0;
    stopReason = ''; % Reason for ending the simulation
    stimEndTimes = zeros(1, N); % Все стимулы изначально не активны
    previousDistanceProgress = -1;
    
    % regularPercentage = 0;  % % аксонов с паттерном regular
    % irregularPercentage = 100;  % % аксонов с паттерном irregular
    % semiregularPercentage = 0;  % % аксонов с паттерном semiregular

    regularPercentage = 64;  % % аксонов с паттерном regular
    irregularPercentage = 18;  % % аксонов с паттерном irregular
    semiregularPercentage = 18;  % % аксонов с паттерном semiregular

    % Количество аксонов для каждого паттерна
    numRegular = round(regularPercentage / 100 * numel(active_axons));
    numIrregular = round(irregularPercentage / 100 * numel(active_axons));
    numSemiregular = round(semiregularPercentage / 100 * numel(active_axons));
    
    % % Смешиваем аксонов и присваиваем их паттернам
    % active_axons_shuffled = active_axons(randperm(length(active_axons)));  % Перемешиваем список активных аксонов
    % % Назначаем аксонов для каждого паттерна
    % regular_axons = active_axons_shuffled(1:numRegular);
    % irregular_axons = active_axons_shuffled(numRegular + 1:numRegular + numIrregular);
    % semiregular_axons = active_axons_shuffled(numRegular + numIrregular + 1:end);
    
    % Назначаем аксонов для каждого паттерна
    regular_axons = active_axons(1:numRegular);
    irregular_axons = active_axons(numRegular + 1:numRegular + numIrregular);
    semiregular_axons = active_axons(numRegular + numIrregular + 1:end);

    % Генерация временных меток стимуляции в зависимости от типа паттерна
    stimTimes = cell(1, N);
    nextStimIdx = ones(1, N);
    
    spikePatterns = cell(1, N);  % Массив для хранения типов паттернов для каждого аксона
    
    for idx = 1:numel(active_axons)
        axon = active_axons(idx);
        ton = stimDelayInterval(1) + (stimDelayInterval(2) - stimDelayInterval(1)) * rand;
        ton_ms = ton; % начальная задержка для аксона
    
        % Определяем тип паттерна для текущего аксона
        if ismember(axon, regular_axons)
            spikePatternType = 'Regular';
        elseif ismember(axon, irregular_axons)
            spikePatternType = 'Irregular';
        else
            spikePatternType = 'Semiregular';
        end
        % Сохраняем паттерн для каждого аксона
        spikePatterns{axon} = spikePatternType;
    
        switch spikePatternType
            case 'Regular'
                k_reg = 500;
                theta_reg = meanISI / k_reg;
                ISI = gamrnd(k_reg, theta_reg, [1, round(maxTime / meanISI)]);
    
                %  Первый стимул = ton_ms, остальные -> кумулятивная сумма ISI
                stimTimes{axon} = [ton_ms, ton_ms + cumsum(ISI(1:end-1))];
    
            case 'Irregular'
                k_irreg = 1;
                theta_irreg = meanISI / k_irreg;
                ISI = gamrnd(k_irreg, theta_irreg, [1, round(maxTime / meanISI)]);
    
                % Генерация временных меток стимуляции
                stimTimes{axon} = [ton_ms, ton_ms + cumsum(ISI(1:end-1))];
    
            case 'Semiregular'
                k_reg = 500;
                theta_reg = meanISI / k_reg;
                ISI = gamrnd(k_reg, theta_reg, [1, round(maxTime / meanISI)]);
    
                % Генерация временных меток стимуляции
                fullSpikeTimes = ton_ms + cumsum(ISI); % Полный список времен стимулов
    
                % Удаляем случайные 30% времен стимулов
                removeIdx = randperm(length(fullSpikeTimes), round(0.3 * length(fullSpikeTimes)));
                fullSpikeTimes(removeIdx) = []; % Удаляем случайные стимулы по всей временной шкале
    
                % Пересчитываем ISI после удаления (новые интервалы между оставшимися спайками)
                ISI = diff(fullSpikeTimes); 
    
                % Финальная генерация стимулов
                stimTimes{axon} = [ton_ms, fullSpikeTimes];
        end

        stimTimes{axon}(stimTimes{axon} <= 0) = [];  % Удаляем стимулы в нуле
    
    end

    % Данные для графиков
    spikeTimesStart = cell(N,1); % Для начала аксона
    spikeTimesMiddle = cell(N,1); % Для середины аксона
    spikeTimesEnd = cell(N,1); % Для конца аксона
    
    % Флаги состояния для каждого узла аксона
    isSpikingStart = false(N, 1); % Для начала аксона
    isSpikingMiddle = false(N, 1); % Для середины аксона
    isSpikingEnd = false(N, 1); % Для конца аксона
    
    % Флаги: достигли ли спайки конца аксона
    spikesReachedEnd = false(1, numel(active_axons));
end

disp(['Максимальное время: ', num2str(maxTime), ' ms; Длина аксона: ', num2str(X), ' мм']);
disp(['Mean_ISI: ', num2str(meanISI), ' мс']);
disp(['Regular pattern: ', num2str(numRegular), ' axons']);
disp(['Irregular pattern: ', num2str(numIrregular), ' axons']);
disp(['Semiregular pattern: ', num2str(numSemiregular), ' axons']);


r = 1+0.*[0:N-1]; % гомогенность диаметров

g = 0.6; % g-ratio
tau = 0.47; % cable time constant ms
taun = 0.03; % node time constant ms
rm = 130; % longitudinal myelin resistance in M\Omega cm
lambc = 1.93.*r.*sqrt(-log(g)); % cable length constant in mm;
lambn = 55e-3.*sqrt(r); % node length constant in mm

% fibre density is chosen at ρ ∈ {0.5 0.6 0.7 0.8 0.9}
rho = 0.6;

sigrat = 1/3;

% define node positions
dn = round(0.2*r./dx); % assuming internode length = 200 x radius
sn = 5; % distance of extremal nodes from domain edge
nn = ceil((X-2*sn)./dx./dn); % number of nodes on each axon
mask = zeros(N,numel(x)); % setup mask (определяем узлы Ранвье)
idn = cell(N,1);
track = cell(N,1);
for o = 1:N
    mask(o,sn/dx+[0:nn(o)-1].*dn(o))=1;
    mask(o,max(size(mask))-sn/dx)=1; 
    idn{o} = sn/dx+[0:nn(o)-1].*dn(o); 
    track{o} = zeros(1,numel(idn{o}));
end
trackn = zeros(N,1);
Vtrack = 40; % пороговое значение потенциала для отслеживания возбуждения

% parameters:
lamb1 = lambc;
deltaL = l/dx;
lamb2 = ((1-deltaL).*(1./lambc.^2) + deltaL.*(1./lambn.^2)).^(-1/2);
tau1 = tau;
tau2 = lamb2.^2.*((1-deltaL).*(tau./lambc.^2) + deltaL.*(taun./lambn.^2));

% dt = 5e-5; % in ms
T = 2e1; % in ms
T = round(T./dt);

% % define system variable:
% V = zeros(N,numel(x));
% I = zeros(N,numel(x));

tau = tau1.*ones(N,numel(x));
lamb = lamb1'*ones(1,numel(x));

for o = 1:N
    tau(o,mask(o,:)==1) = tau2(o);
    lamb(o,mask(o,:)==1) = lamb2(o);
end

% HH stuff:
Vn = zeros(N,max(sum(mask,2))+1);
m = 0.0529.*ones(size(Vn));
h = 0.5961.*ones(size(Vn));
n = 0.3177.*ones(size(Vn));
Inode = zeros(size(Vn));

% HH parameters: (2nd row: Brill et al. '77)
gNa = 120; gNa = 1200; gNa = 4800;
eNa = 115;
gK = 36; gK = 90; gK = 720;
eK = -12;
gL = 0.3; gL = 20; gL = 30;

%while max(trackn' - nn) < 0 && t * dt < maxTime
%while ~all(spikesReachedEnd) && t * dt < maxTime
while t * dt < maxTime
    t = t+1;
    
    Istim = zeros(1, N);
    for idx = 1:numel(active_axons)
        axon = active_axons(idx);

        % Проверяем, активен ли текущий стимул
        if t * dt < stimEndTimes(axon)
            Istim(axon) = r(axon) * 5e3; % Стимуляция продолжается
        elseif nextStimIdx(axon) <= length(stimTimes{axon}) && ...
               stimTimes{axon}(nextStimIdx(axon)) <= t * dt
            Istim(axon) = r(axon) * 5e3; % Новый стимул
            stimEndTimes(axon) = t * dt + 2.5; % Устанавливаем время окончания стимула
            nextStimIdx(axon) = nextStimIdx(axon) + 1;
        end
    end

    for o = 1:N
        Vn(o,1:nn(o)+1) = V(o,mask(o,:)==1);
    end

    % Hodgkin-Huxley stuff
    m = m + dt.*(((2.5-0.1.*Vn) ./ (exp(2.5-0.1.*Vn) -1)).*(1-m) - ...
        (4.*exp(-Vn./18)).*m);
    h = h + dt.*(((0.07.*exp(-Vn./20))).*(1-h) - ...
        (1./(exp(3.0-0.1.*Vn)+1)).*h);
    n = n + dt.*(((0.1-0.01.*Vn) ./ (exp(1-0.1.*Vn) -1)).*(1-n) - ...
        (0.125.*exp(-Vn./80)).*n);

    Inode = repmat(r',[1 max(size(Inode))]).*(gNa.*m.^3.*h.*(eNa-Vn) + gK.*n.^4.*(eK-Vn))./gL;
    
    Inode(:,1) = Istim; % стимул прикладывается к первому узлу Ранвье

    % % стимул прикладывается к среднему узлу Ранвье
    % for o = 1:N
    %     midNode = floor(nn(o)/2);  % индекс середины (в idn)
    %     Inode(o, midNode) = Istim(o);
    % end

    for o = 1:N
        I(o,mask(o,:)==1) = Inode(o,1:nn(o)+1);
    end

    Ve = [V(:,1) V V(:,end)];
    Vxx = (Ve(:,3:end)+Ve(:,1:end-2)-2.*Ve(:,2:end-1))./dx^2;

    dV = dt.*(lamb.^2.*Vxx - lamb.^2.*repmat(sum( repmat( (r'.^2./sum(r.^2))./(1+sigrat*((1-rho)/(g^2*rho))) ,[1 numel(x)]).*Vxx,1),[N 1]) - V + I)./tau;
    V = V + dV;

    % Фиксация спайков
    for o = 1:N
        % Индексы узлов Ранвье
        nodeStart = idn{o}(2); % Второй узел
        nodeMiddle = idn{o}(floor(nn(o)/2)); % Средний узел
        nodeEnd = idn{o}(end); % Последний узел

        % Проверка порогового значения потенциала
        % Начало аксона
        if V(o,nodeStart) > Vtrack
            if ~isSpikingStart(o) % Если спайк еще не зафиксирован
                spikeTimesStart{o} = [spikeTimesStart{o}, t * dt];
                isSpikingStart(o) = true; % Устанавливаем флаг спайка
            end
        else
            isSpikingStart(o) = false; % Сбрасываем флаг, если потенциал опустился ниже порога
        end
    
        % Середина аксона
        if V(o,nodeMiddle) > Vtrack
            if ~isSpikingMiddle(o)
                spikeTimesMiddle{o} = [spikeTimesMiddle{o}, t * dt];
                isSpikingMiddle(o) = true;
            end
        else
            isSpikingMiddle(o) = false;
        end
    
        % Конец аксона
        if V(o,nodeEnd) > Vtrack
            if ~isSpikingEnd(o)
                spikeTimesEnd{o} = [spikeTimesEnd{o}, t * dt];
                isSpikingEnd(o) = true;
            end
        else
            isSpikingEnd(o) = false;
        end

    end

    for p = 1:N
        if trackn(p)<nn(p)
            if sum(V(p,idn{p}(trackn(p)+1:end))>Vtrack)>0 % если в следующем узле аксона p потенциал превысил порог
                idt = max(find(V(p,idn{p}(trackn(p)+1:end))>Vtrack));
                trackn(p) = trackn(p) + idt;  % обновляем индекс последнего возбужденного узла
            end
        end
    end

    for idx = 1:numel(active_axons)
        axon = active_axons(idx);
        if trackn(axon) >= nn(axon)
            spikesReachedEnd(idx) = true;
        end
    end

    % Progress tracking
    totalTime = t * dt;
    timeProgress = (totalTime / maxTime) * 100;
    [~, maxTrackIdx] = max(trackn);
    interNodeDist = X / (nn(maxTrackIdx) - 1);
    totalDistance = (max(trackn) - 1) * interNodeDist;
    distanceProgress = (totalDistance / X) * 100;
    % if mod(timeProgress, 5) == 0 || mod(distanceProgress, 5) == 0
    %     if distanceProgress ~= previousDistanceProgress
    %         fprintf('Time: %.2f ms, Time progress: %.2f%%, Distance progress: %.2f mm (%.2f%% of axon length)\n', ...
    %                 totalTime, timeProgress, totalDistance, distanceProgress);
    %         previousDistanceProgress = distanceProgress;
    %     end
    % end
    if mod(timeProgress, 5) == 0
        fprintf('Time: %.2f ms, Time progress: %.2f%%, Distance progress: %.2f mm (%.2f%% of axon length)\n', ...
                totalTime, timeProgress, totalDistance, distanceProgress);
    end

    % create checkpoint
    if mod(t * dt, checkpointStep) == 0 
        save(checkpointFileName, ...
             'X', 'dx', 'l', 'x', 'maxTime', 'N', 'active_axons', 'meanISI', ...
             'stimTimes', 'nextStimIdx', 'spikePatterns', ...
             'V', 'I', 't', 'dt', 'totalTime', ...
             'stopReason', 'stimEndTimes', 'previousDistanceProgress', ...
             'spikeTimesStart', 'spikeTimesMiddle', 'spikeTimesEnd', ...
             'isSpikingStart', 'isSpikingMiddle', 'isSpikingEnd', ...
             'spikesReachedEnd');
        disp(['Checkpoint saved at t=', num2str(t * dt), ' ms']);
    end

    % Очистка памяти
    if mod(t, 1000) == 0
        clear Ve Vxx dV;
    end

    % Check stop condition
    % if max(trackn' - nn) >= 0
    %     stopReason = 'End of axon reached';
    %     break;
    
    % if all(spikesReachedEnd)
    %     stopReason = 'End of active axons reached';
    %     break;
    % elseif t * dt >= maxTime
    %     stopReason = 'Max time reached';
    %     break;
    % end

    if t * dt >= maxTime
        stopReason = 'Max time reached';
        break;
    end
end

disp(['Simulation ended because: ', stopReason]);

% Сохранение данных
save('SpikeData.mat', 'spikeTimesStart', 'spikeTimesMiddle', 'spikeTimesEnd', 'totalTime', 'spikePatterns', 'X');
disp('Данные о временах спайков сохранены в файл SpikeData.mat');

elapsedTime = toc; % Конец измерения времени
endTime = now; % фиксируем момент окончания
disp(['Конец расчета: ', datestr(endTime, 'dd-mm-yyyy HH:MM:SS')]);
disp(['Длительность расчета: ', num2str(elapsedTime), ' секунд']);

% END OF FILE
