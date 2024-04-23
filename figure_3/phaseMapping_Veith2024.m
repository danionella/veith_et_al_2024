clear all, close all, clc

% This Matlab script processes velocity data obtained from motionDetect_Veith2024.m 
% (using PIVlab toolbox, version 2.31), as described in Veith et al. 2024. 
% The basic principle is described in Fig.3 and Extended data Fig. 10. 

% The current script performs the following steps:
% 1. Load velocity maps 
% 2. Compute first Fourier component
% 4. Save data to be used with phaseMapping_Veith2024.m

% Version: 1.0, 01 April 2024 
% Author:  Johannes Veith, Thomas Chaigne, Benjamin Judkewitz


%% Load PIV vectors and acquisition parameters 
[resFile,filepath]=uigetfile(mfilename('fullpath')); % load motion data ('cmdPIV[...].mat')
load([filepath resFile]);
str = strsplit(filepath,filesep);
day = erase(str{end-2},'_processing');
time = str{end-1}(4:8);
load([strjoin(str(1:end-3),filesep) filesep day filesep str{end-1} filesep 'acqParam.mat'])

%% Define relevant acquisition parameters
otherdims = repmat({':'},1,ndims(uu)-3); % e.g. phases, amps, depth, use as otherdims{:}
if NimgTot==2*Nimg %condition of one image with playback and one without
    silenceInd=1:(Nimg*nStep); %as long as delay1=duration
    noiseInd=(Nimg*nStep+1):2*(Nimg*nStep);
end
if NimgTot==1*Nimg
    silenceInd=1:-1; %empty
    noiseInd=1:Nimg*nStep;
end

lineRate=rigAIrate/config.scancfg.nInSamplesPerLine;
frameRate=lineRate/config.scancfg.nLinesPerFrame;
linePeriod=1/lineRate;
zoomFactor=config.scancfg.zoom;
fillFraction=config.scancfg.fillFraction; 

%% Extract first Fourier component

% Places the nStep stoboscopic measurements in first dimension
uu1 = permute(uu,[3 1 2 4:length(size(squeeze(uu)))]);
vv1 = permute(vv,[3 1 2 4:length(size(squeeze(vv)))]);

% Fourier transform to get phase and amplitude of motion at acoustic stimulation frequency 
uu2 = fft(uu1(noiseInd,:,:,otherdims{1:end}), [], 1);
vv2 = fft(vv1(noiseInd,:,:,otherdims{1:end}), [], 1);

% Now last dimension stores the nStep fourier modes (index=2 is the first AC component)
uu3 = permute(uu2, [2 3 4:length(size(squeeze(uu))) 1]); 
vv3 = permute(vv2, [2 3 4:length(size(squeeze(vv))) 1]);

% Correct for phase accumulating while scanning (see Extended Data Fig.11 for more detail)
[X,Y]=meshgrid(1:size(uu,2),1:size(uu,1));
linePeriod2=linePeriod*fillFraction;
phaseScan=rem((linePeriod2/size(uu,2)*X)*freq,1)*2*pi ;
uu4=exp(-1i*phaseScan).*uu3;
vv4=exp(-1i*phaseScan).*vv3;

% Sanity check
if rem(freq/lineRate-1/nStep,1)~=0
    error('Frequencies should match')
end

% FT needs to be divided by number of samples and multiplied by 2 to get the amplitude of the
% single-sided spectrum. (See https://fr.mathworks.com/help/matlab/ref/fft.html for more detail).
uu5 = 2*uu4/nStep;

% Convert particle velocity from pixel/framePeriod to pixel/s
% --> Virtual frame period is nStep*freq (see Extended Data Fig.11b for more
% detail). 
uu6 = nStep*freq*uu5; 
vv6 = nStep*freq*vv5;

clear uu1 uu2 uu3 uu4 uu5

%% Use calibrated field-of-view size to get absolute displacement values
lensFOV_um = 279.2489/config.scancfg.zoom; % calibrated
recordedFOV_um =  config.scancfg.fillFraction*lensFOV_um;
umPerpx = recordedFOV_um/config.scancfg.nPixelsPerLine;

% Convert particle velocity from pixel/s to um/s
u_um = umPerpx*uu6;
v_um = umPerpx*vv6;

% Compute displacement amplitude
x_um = u_um/(2*pi*freq); % x = A * exp(iwt) --> x = v / iw
y_um = v_um/(2*pi*freq);

fcomponent = 2; % 1st AC=2

%% Select parameters depending on dataset to analyse - figure to reproduce
% in Veith et al. 2024 Fig.3 - Ext Data Fig.11

% select motion axis 
take_axis = 'x'; % 'x' top row, 'y' bottom row

% select stimulation configuration
stimFlag = 'pressure'; % 'single speaker', 'pressure', 'particle motion'

switch stimFlag
    case 'single speaker'
        % left column - single speaker
        kk = 1; % phase
        kkk= 1; % amp

    case 'pressure'
        % middle column - in-phase speakers (pressure)
        kk = 1; % phase
        kkk= 2; % amp

    case 'particle motion'
        % right column - anti-phase speakers (particle motion)
        kk = 3; % phase
        kkk= 2; % amp

    otherwise 
        error('The requested condition does not exist.')
end

switch [day filesep time]
   case '200930/12h03'
        % 3c - 11a - 200930/12h03
        kkkkstart= 1;
        kkkkstop= 8;
        % set constant scale (referred to as Amax in Fig.)
        lim = 8e3; % 3e3 or 8e3, in µm/s, constant limits across conditions for visual comparison

    case '200814/15h19'
        % 3d - 11b - 200814/15h19
        kkkkstart= 2; %1 %2 %1 %depth
        kkkkstop= 5;%10 %5 %8 size(img3,6);
        % set constant scale (referred to as Amax in Fig.)
        lim = 3e3; % 3e3 or 8e3, in µm/s, constant limits across conditions for visual comparison

    case '200930/14h21'
        % 3e - 11c - 200930/14h21
        kkkkstart= 1;
        kkkkstop= 10;
        % set constant scale (referred to as Amax in Fig.)
        lim = 8e3; % 3e3 or 8e3, in µm/s, constant limits across conditions for visual comparison

    otherwise
        error('The dataset is not recognised.')
end

%% Compute phase map maximum intensity projection 

if take_axis == 'x'
    cim = max(u_um(:,:,kk,kkk,kkkkstart:kkkkstop,fcomponent),[],5,"omitnan"); 
elseif take_axis == 'y'
    cim = max(v_um(:,:,kk,kkk,kkkkstart:kkkkstop,fcomponent),[],5,"omitnan");
end

% Single speaker condition had half speaker power, rescale
if kkk==1 
    cim = 2 * cim;
end

% Rescale image to size of background
out = cimage(cim,lim,'hsv');
out(isnan(out))=0; % nan color
% Upscale to fit background, be aware of PIV discretization when scaling, introduces small error
out = imresize(out, size(bckgrnd)); 
[outh outw c] = size(out);
out = rot90(imresize(out, [scaleFactorY*outh outw]),-1); % square FOV

% Color scale plot
[re, im] = meshgrid(linspace(-lim,lim,500));
c = re + 1i*im;
out_scale = cimage(c, lim);

% Save figures
processingFolderFigures = [filepath 'figures'];
cfname1 = 'phaseMap_maxProj.png'; cfpath1 = strjoin([processingFolderFigures filesep str{end-1} '_amp' string(kkk) 'phase' string(kk) '_planes' string(kkkkstart) '-' string(kkkkstop) '_axis' take_axis '_lim' lim 'um-s_' cfname1],'');
cfname2 = 'phaseMap_maxProj_scale.png'; cfpath2 = [processingFolderFigures filesep str{end-1} '_' cfname2];

imwrite(out,cfpath1,'png');
imwrite(out_scale,cfpath2,'png');

return

%% Bonus section for Fig.3e: This part is only used to compute average displacement amplitude and phase in given regions of interest 
% as in Fig.3d. It can be edited to analyse further other data.
% ROIs used in Veith et al. 2024 are stored in Veith_et_al_2024/200930_processing/Acq14h21_doubleSineShifted_phaseNamp1000line2line_zoom0397_4096x1024_unidir/masks

filename = str{end-1};

% Set to 1 to draw ROI on first usage:
draw_roi = 0;

% Scale just for display
scale = 0.3;
maxim = abs(max(cim(:)));
lim=scale*maxim;
out=cimage(cim,lim);
out(isnan(out)) = 0;

processingFolderMasks = [filepath 'masks'];
mkdir(processingFolderMasks);
cfname1 = 'ROI1'; cfpath1 = [processingFolderMasks filesep filename '_' cfname1];
cfname2 = 'ROI2'; cfpath2 = [processingFolderMasks filesep filename '_' cfname2];

if draw_roi
    % Select ROIs
    ROI1 = roipoly(out); disp(sum(ROI1(:)));
    ROI2 = roipoly(out); disp(sum(ROI2(:)));
    ROI12 = logical(ROI1 + ROI2);
    % Save
    save(cfpath1,'ROI1');
    save(cfpath2,'ROI2');
else
    ROI1 = load(cfpath1).ROI1;
    ROI2 = load(cfpath2).ROI2;
end
ROI12 = logical(ROI1 + ROI2);
figure(1), imshow(0.5*out+0.5*ROI12);

imwrite(0.7*out+0.3*ROI1,strjoin([cfpath1 'amp' string(kkk) 'phase' string(kk) '_planes' string(kkkkstart) '-' string(kkkkstop) '_axis' take_axis '.png'],''),'png');
imwrite(0.7*out+0.3*ROI2,strjoin([cfpath2 'amp' string(kkk) 'phase' string(kk) '_planes' string(kkkkstart) '-' string(kkkkstop) '_axis' take_axis '.png'],''),'png');

% Store ROI values in structure
sROI.freq = freq;
sROI.filename = filename;
sROI.day = day;
sROI.take_axis = take_axis;

% Compute displacement amplitude in ROI
sROI.mean1 = mean(abs(cim(ROI1)),'omitnan')/(2*pi*freq);
sROI.std1 = std(abs(cim(ROI1)),'omitnan')/(2*pi*freq);
sROI.mean2 = mean(abs(cim(ROI2)),'omitnan')/(2*pi*freq);
sROI.std2 = std(abs(cim(ROI2)),'omitnan')/(2*pi*freq);
sROI.unit = 'um';

% Compute phase in ROI
angleROI1 = angle(cim(ROI1));
angleROI1_nonan = angleROI1(~isnan(angleROI1));
sROI.radians_mean1 = circ_mean(angleROI1_nonan); % requires Circular Statistics Toolbox (Directional Statistics)
sROI.radians_std1 = circ_std(angleROI1_nonan);
sROI.degrees_mean1 = 180*sROI.radians_mean1/pi;
sROI.degrees_std1 = 180*sROI.radians_std1/pi;

angleROI2 = angle(cim(ROI2));
angleROI2_nonan = angleROI2(~isnan(angleROI2));
sROI.radians_mean2 = circ_mean(angleROI2_nonan); 
sROI.radians_std2 = circ_std(angleROI2_nonan);
sROI.degrees_mean2 = 180*sROI.radians_mean2/pi;
sROI.degrees_std2 = 180*sROI.radians_std2/pi;

% Plot 
figure(11), subplot(121)
errorbar(1,sROI.mean1,sROI.std1,'LineWidth', 2),hold on
errorbar(2,sROI.mean2,sROI.std2,'LineWidth', 2)
hold off
xlim([0 3]), ylim([0 2])
title(['Displacement amplitude along ' num2str(sROI.take_axis) ' in ' sROI.unit])
xlabel('ROI #')

offset = pi;
subplot(122)
errorbar(1,(sROI.radians_mean1+offset)/pi,sROI.radians_std1/pi,'LineWidth', 2),hold on
errorbar(2,(sROI.radians_mean2+offset)/pi,sROI.radians_std2/pi,'LineWidth', 2)
hold off
xlim([0 3])
ylim([-1.1+offset/pi 1.1+offset/pi])
title('Phase (in multiple of π)')
xlabel('ROI #')
