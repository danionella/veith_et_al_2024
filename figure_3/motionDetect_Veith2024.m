clear all, close all, clc

% This Matlab script processes vibrometry data acquired from raster-based
% vibrometry as described in Veith et al. 2024. The basic principle is
% described in Fig.3 and Extended data Fig. 10. The vibrometry microscope
% is driven by this custom software: https://github.com/danionella/lsmaq

% The current script performs the following steps:
% 1. Load data
% 2. Reshape image series to reconstruct movie of vibrating structures
% 3. Compute particle motion amplitude/velocity using PIVlab (tested with
% version 2.31).
% 4. Save data to be used with phaseMapping_Veith2024.m

% Version: 1.0, 01 April 2024 
% Author:  Johannes Veith, Thomas Chaigne, Benjamin Judkewitz

%% Load data

% Select the folder where data are stored
dataDir = uigetdir(mfilename('fullpath'));

[filepath,filename,ext] = fileparts(dataDir);
main = filepath(1:end-6);
day = [filepath(end-5:end) filesep];
time = filename(4:8);

% Load acquisition parameters and define variables
load([dataDir filesep 'acqParam.mat'])

% Load data
mf = matfile([dataDir filesep 'acqWphaseShift0001.mat']);
sz = size(mf.data);

% Flatten and permute
resh = sz~=1;
resh(4) = 1; %keep the fourth dimension even if just one image was taken
img = permute(reshape(mf.data,sz(resh)),[2 1 3:length(sz)]);  %flatten the other dims

% Dimensions of img:
% 1: vertical columns
% 2: horizontal lines
% 3: sound off/on, only for some datasets (200814 - 15h19)
% 4: phase between left and right speakers (phaseShift)
% 5: sound amplitude (amp)
% 6: depth (zStart:zStepSize:zEnd)

% Load more acquisition parameters (lsmaq metadata), and then compute some more parameters
config=mf.config;
lineRate=rigAIrate/config.scancfg.nInSamplesPerLine;
frameRate=lineRate/config.scancfg.nLinesPerFrame;
linePeriod=1/lineRate;
zoomFactor=config.scancfg.zoom;
fillFraction=config.scancfg.fillFraction;

% Get useful handles for indices
if NimgTot==2*Nimg % condition of one image with playback and one without (only for 200814 - 15h19)
    silenceInd=1:(Nimg*nStep);
    noiseInd=(Nimg*nStep+1):2*(Nimg*nStep);
end

if NimgTot==1*Nimg
    silenceInd=1:-1; %empty
    noiseInd=1:Nimg*nStep;
end

otherdims = repmat({':'},1,ndims(img)-3); % e.g. phases, amps, depth, use as otherdims{:}

%% Interleave images to reconstruct movie of moving structures

sz=size(img);
if NimgTot==2*Nimg
    img3silence = permute(reshape(img(:,:,1:Nimg,:), [nStep sz(1)/nStep sz(2) Nimg sz(4:end)]), [2 3 1 4:(length(sz)+1)]);
    img3noise = permute(reshape(img(:,:,1+Nimg:2*Nimg,:), [nStep sz(1)/nStep sz(2) Nimg sz(4:end)]), [2 3 1 4:(length(sz)+1)]);
    img3=cat(3,img3silence,img3noise);
    clear img3silence img3noise
end
if NimgTot==1*Nimg
    img3 = permute(reshape(img(:,:,1:Nimg,:), [nStep sz(1)/nStep sz(2) Nimg sz(4:end)]), [2 3 1 4:(length(sz)+1)]);
end

img3=squeeze(img3);

% Dimensions of img3:
% 1: vertical columns
% 2: horizontal lines
% 3: nStep images, with 2pi/nStep sound phases steps
% 4: phase between left and right speakers (phaseShift)
% 5: sound amplitude (amp)
% 6: depth (zStart:zStepSize:zEnd)

scaleFactorY=size(img3,2)/size(img3,1); % the field-of-view is square, each line is scanned nStep times
imgsz = size(img3,[1,2]);

%% %%%%%%% Motion detection with PIVlab %%%%%%
% tested with PIVlab 2.31; for 3.0 additional settings need to be defined

%% PIVlab cmd: settings

sz=size(img3);
amount = prod(sz(3:end));
disp(['Found ' num2str(amount) ' images.'])

% PIV Settings
s = cell(11,2);
%Parameter                          %Setting           %Options
s{1,1}= 'Int. area 1';              s{1,2}=32;         % window size of first pass
s{2,1}= 'Step size 1';              s{2,2}=16;         % step of first pass
s{3,1}= 'Subpix. finder';           s{3,2}=1;          % 1 = 3point Gauss, 2 = 2D Gauss
s{4,1}= 'Mask';                     s{4,2}=[];         % If needed, generate via: imagesc(image); [temp,Mask{1,1},Mask{1,2}]=roipoly;
s{5,1}= 'ROI';                      s{5,2}=[];         % Region of interest: [x,y,width,height] in pixels, may be left empty
s{6,1}= 'Nr. of passes';            s{6,2}=2;          % 1-4 nr. of passes
s{7,1}= 'Int. area 2';              s{7,2}=4;          % second pass window size
s{8,1}= 'Int. area 3';              s{8,2}=4;          % third pass window size
s{9,1}= 'Int. area 4';              s{9,2}=2;          % fourth pass window size
s{10,1}='Window deformation';       s{10,2}='*linear'; % '*spline' is more accurate, but slower
s{11,1}='Repeated Correlation';     s{11,2}=0;         % 0 or 1 : Repeat the correlation four times and multiply the correlation matrices.
s{12,1}='Disable Autocorrelation';  s{12,2}=0;         % 0 or 1 : Disable Autocorrelation in the first pass.
s{13,1}='Correlation style';        s{13,2}=0;         % 0 or 1 : Use circular correlation (0) or linear correlation (1).

%% Image preprocessing settings

p = cell(8,1);

%Parameter                       %Setting           %Options
p{1,1}= 'ROI';                   p{1,2}=s{5,2};     % same as in PIV settings
p{2,1}= 'CLAHE';                 p{2,2}=1;          % 1 = enable CLAHE (contrast enhancement), 0 = disable
p{3,1}= 'CLAHE size';            p{3,2}=500;        % CLAHE window size
p{4,1}= 'Highpass';              p{4,2}=0;          % 1 = enable highpass, 0 = disable
p{5,1}= 'Highpass size';         p{5,2}=15;         % highpass size
p{6,1}= 'Clipping';              p{6,2}=0;          % 1 = enable clipping, 0 = disable
p{7,1}= 'Wiener';                p{7,2}=0;          % 1 = enable Wiener2 adaptive denaoise filter, 0 = disable
p{8,1}= 'Wiener size';           p{8,2}=3;          % Wiener2 window size

% Note: keep int16 to use same values as in PIVlab GUI (if used, see below)

p{9,1}= 'Minimum intensity';     p{9,2}=0;         % Minimum intensity of input image (0 = no change)

switch [day time]
    case '200930/12h03'
        % 3c - 11a - 200930/12h03
        p{10,1}='Maximum intensity';     p{10,2}=0.4;      % Maximum intensity on input image (1 = no change)

    case '200814/15h19'
        % 3d - 11b - 200814/15h19
        p{10,1}='Maximum intensity';     p{10,2}=0.8;      % Maximum intensity on input image (1 = no change)

    case '200930/14h21'
        % 3e - 11c - 200930/14h21
        p{10,1}='Maximum intensity';     p{10,2}=0.6;      % Maximum intensity on input image (1 = no change)

    otherwise
        error('The dataset is not recognised.')
end

% Preprocess images
imageTest = PIVlab_preproc(uint16(img3(:,:,1,1,1,3)),p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2},p{9,2},p{10,2});
figure(6), imagesc(imageTest), axis image

%% PIVlab cmd: run

wbar=waitbar(0,'Analyzing...');
x=cell(amount,1);
y=x;
u_original=x;
v_original=x;
typevector=x; % typevector will be 1 for regular vectors, 0 for masked areas
data=squeeze(img3);
counter=0;

% what phase is compared to what other phase:
kto_list = zeros(size(data,3),size(data,4),size(data,5),size(data,6));

for kkkk=1:size(data,6) % depth
    disp(kkkk)
    for kkk=1:size(data,5) % amp
        for kk=1:size(data,4) % phase
            for k=1:size(data,3) % ~time
                counter=counter+1;

                % rolling: consecutive images
                meth='rolling';
                kto = floor((k-1)/nStep)*nStep + rem(k,nStep) + 1; % detect motion from e.g. 4th image to first as well (close one period)
                kto_list(k,kk,kkk,kkkk)=kto;

                image1=uint16(data(:,:,k,kk,kkk,kkkk)); % reference image
                image2=uint16(data(:,:,kto,kk,kkk,kkkk));

                image1 = PIVlab_preproc(image1,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2},p{9,2},p{10,2}); %preprocess images
                image2 = PIVlab_preproc(image2,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2},p{9,2},p{10,2});

                [x{counter}, y{counter}, u_original{counter}, v_original{counter}, typevector{counter}] = ...
                    piv_FFTmulti (image1,image2,s{1,2},s{2,2},s{3,2},s{4,2},s{5,2},s{6,2},s{7,2},s{8,2},s{9,2},s{10,2},s{11,2},s{12,2},s{13,2});

                waitbar((counter)/amount,wbar,sprintf('Analyzing... Comparing image %i to image %i',kto,k))

                % Graphical output (disable to improve speed)
                %{
                imagesc(double(image1)+double(image2));colormap('gray');
                hold on
                quiver(x{counter},y{counter},u_original{counter},v_original{counter},'g','AutoScaleFactor', 1.5);
                hold off;
                axis image;
                title(['amp: ', num2str(kkk), ', phase: ', num2str(kk), ', timestep: ',num2str(k), ' to ', num2str(kto)],'interpreter','none')
                set(gca,'xtick',[],'ytick',[])
                daspect([scaleFactorY 1 1])
                drawnow;
                %}
            end
        end
    end
end
waitbar((counter)/amount,wbar,sprintf('Done'))


%% PIVlab cmd: reshape PIV results to match img3 dimensions

xx=cell2mat(x);
szp=size(x{1});
xx=permute(reshape(xx.',szp(2),szp(1),length(x)),[2 1 3]);
xx=reshape(xx,[szp(1),szp(2),sz(3:end)]);

uu=cell2mat(u_original);
szp=size(u_original{1});
uu=permute(reshape(uu.',szp(2),szp(1),length(u_original)),[2 1 3]);
uu=reshape(uu,[szp(1),szp(2),sz(3:end)]);

yy=cell2mat(y);
szp=size(y{1});
yy=permute(reshape(yy.',szp(2),szp(1),length(y)),[2 1 3]);
yy=reshape(yy,[szp(1),szp(2),sz(3:end)]);

vv=cell2mat(v_original);
szp=size(v_original{1});
vv=permute(reshape(vv.',szp(2),szp(1),length(v_original)),[2 1 3]);
vv=reshape(vv,[szp(1),szp(2),sz(3:end)]);

%% Compute Maximum Intensity Projections for figure
% Select parameters depeding on dataset to analyse - figure to reproduce Fig.3 - Ext Data Fig.11

switch [day time]

    case '200930/12h03'
        % 3c - 11a - 200930/12h03
        kkkkstart= 1;
        kkkkstop= 8;

    case '200814/15h19'
        % 3d - 11b - 200814/15h19
        kkkkstart= 2; %1 %2 %1 %depth
        kkkkstop= 5;%10 %5 %8 size(img3,6);

    case '200930/14h21'
        % 3e - 11c - 200930/14h21
        kkkkstart= 1;
        kkkkstop= 10;

    otherwise
        error('The dataset is not recognised.')
end

if length(silenceInd)>0
    bckgrnd = max(double(img3(:,:,silenceInd(end),1,1,kkkkstart:kkkkstop)),[],6);
else
    bckgrnd = max(double(img3(:,:,1,1,1,kkkkstart:kkkkstop)),[],6);
end
bckgrnd = imadjust(bckgrnd/max(bckgrnd(:)),[0 .6 ],[]);
bckgrnd = rot90(imresize(bckgrnd, [scaleFactorY*size(img3,1) size(img3,2)]),-1); % square FOV, rotate to match Fig. orientation
figure(1), imagesc(bckgrnd), axis image, set(gca,'visible','off'), colormap gray

%% Save results & return cmdPIV.mat (motion vectors, their positions and PIV parameters)
% save representative maximum intensity projection as well (bckgrnd)

% PIV results directory
processingFolder = [main day(1:end-1) '_processing' filesep filename];
mkdir(processingFolder);

% Figure directory
processingFolderFigures = [processingFolder filesep 'figures'];
mkdir(processingFolderFigures);

param=['_' meth '_Pass'];
passSize=[s{1,2},s{7,2},s{8,2},s{9,2}];
for kpass=1:s{6,2}
    param=cat(2,param,'_',num2str(passSize(kpass)));
end
param=cat(2,param,'_step1stPass',num2str(s{2,2}));
if p{2,2} % if clahe enabled
    param=cat(2,param,'_Clahe',num2str(p{3,2}));
end
param=cat(2,param,'_minInt',num2str(p{9,2}),'_maxInt',num2str(p{10,2}));
param=erase(param,'.');

% Save PIV results
resName=[processingFolder filesep 'cmdPIV' param '.mat'];
save(resName,'xx','yy','uu','vv','s','p','kto_list','scaleFactorY','config','bckgrnd');

% Save current figure
cfname3 = 'phaseMap_maxProj_bg.png';
cfpath3 = strjoin([processingFolderFigures filesep filename  '_planes' string(kkkkstart) '-' string(kkkkstop) '_' cfname3],'');
imwrite(bckgrnd,cfpath3,'png');

% The next part of the processing is done in phaseMapping_veith2024.m

return

%% Bonus 1: PIVlab postprocessing - not used in Veith et al. 2024

postProcFlag = 0;

if  postProcFlag
    wbar=waitbar(0,'Filtering...');

    % Settings
    amount = prod(sz(3:end));
    velLimFlag=1;
    umin = -10; % minimum allowed u velocity, adjust to your data
    umax = 10; % maximum allowed u velocity, adjust to your data
    vmin = -10; % minimum allowed v velocity, adjust to your data
    vmax = 10; % maximum allowed v velocity, adjust to your data
    stdLimFlag=0;
    stdthresh=7; % threshold for standard deviation check
    medLimFlag=0;
    epsilon=0.15; % epsilon for normalized median test
    thresh=3; % threshold for normalized median test
    interpFlag=1;

    u_filt=cell(amount,1);
    v_filt=u_filt;
    %    typevector_filt=u_filt;

    for PIVresult=1:nStep

        waitbar(PIVresult/amount,wbar,'Filtering...')

        if exist('u_original')
            u_filtered=u_original{PIVresult,1};
            v_filtered=v_original{PIVresult,1};
            %   typevector_filtered=typevector{PIVresult,1};
        elseif exist('uu')
            u_filtered=uu(:,:,PIVresult);
            v_filtered=vv(:,:,PIVresult);
            %   typevector_filtered=uu; % just to avoid errors
        else
            waitbar(PIVresult/amount,wbar,'you got a PB')
        end

        %vellimit check
        if velLimFlag
            u_filtered(u_filtered<umin)=NaN;
            u_filtered(u_filtered>umax)=NaN;
            v_filtered(v_filtered<vmin)=NaN;
            v_filtered(v_filtered>vmax)=NaN;
        end

        % stddev check
        if stdLimFlag
            
            meanu=nanmean(abs(u_filtered), [1 2]);; %modified
            meanv=nanmean(abs(v_filtered), [1 2]);; %modified
            std2u=nanstd(reshape(u_filtered,size(u_filtered,1)*size(u_filtered,2),1));
            std2v=nanstd(reshape(v_filtered,size(v_filtered,1)*size(v_filtered,2),1));
            minvalu=meanu-stdthresh*std2u;
            maxvalu=meanu+stdthresh*std2u;
            minvalv=meanv-stdthresh*std2v;
            maxvalv=meanv+stdthresh*std2v;
         
            u_filtered(abs(u_filtered)<minvalu)=NaN; %modified
            u_filtered(abs(u_filtered)>maxvalu)=NaN; %modified
            v_filtered(abs(v_filtered)<minvalv)=NaN; %modified
            v_filtered(abs(v_filtered)>maxvalv)=NaN; %modified
        end

        % normalized median check
        if medLimFlag
            %Westerweel & Scarano (2005): Universal Outlier detection for PIV data
            [J,I]=size(u_filtered);
            medianres=zeros(J,I);
            normfluct=zeros(J,I,2);
            b=1;
            for c=1:2
                if c==1; velcomp=u_filtered;else;velcomp=v_filtered;end %#ok<*NOSEM>
                for i=1+b:I-b
                    for j=1+b:J-b
                        neigh=velcomp(j-b:j+b,i-b:i+b);
                        neighcol=neigh(:);
                        neighcol2=[neighcol(1:(2*b+1)*b+b);neighcol((2*b+1)*b+b+2:end)];
                        med=median(neighcol2);
                        fluct=velcomp(j,i)-med;
                        res=neighcol2-med;
                        medianres=median(abs(res));
                        normfluct(j,i,c)=abs(fluct/(medianres+epsilon));
                    end
                end
            end
            info1=(sqrt(normfluct(:,:,1).^2+normfluct(:,:,2).^2)>thresh);
            u_filtered(info1==1)=NaN;
            v_filtered(info1==1)=NaN;
        end

        %Interpolate missing data
        if interpFlag
            u_filtered=inpaint_nans(u_filtered,4);
            v_filtered=inpaint_nans(v_filtered,4);
        end

        u_filt{1,PIVresult}=u_filtered; %todo ?
        v_filt{1,PIVresult}=v_filtered;
        % typevector_filt{1,PIVresult}=typevector_filtered;
    end
    waitbar(1,wbar,'You just got post-processed!')
    % clearvars -except p s sz x y u v typevector directory filenames u_filt v_filt typevector_filt
end


%% Bonus 2: use PIVlab with GUI - not used in Veith et al. 2024

GUIflag=0; % use GUI or cmd for PIVlab
% PIVlab can be either used through a GUI, or directly with Matlab commands
% to use the GUI, reshaped images must be saved first in the right format

% save images
if GUIflag

    options.overwrite=1;
    if ~isfile([main day(1:end-1) '_processing' filesep filename '_img3' filesep 'img_1_1_01.tif']) % do not save images of saved already
        data=squeeze(img3);
        for kkkk=1:size(data,6) % depth
            disp(kkkk)
            for kkk=1:size(data,5) % amp
                for kk=1:size(data,4) % phase
                    for k=1:size(data,3) % ~time
                        saveastiff(cast(data(:,:,k,kk,kkk,kkkk),'uint16'),...
                            [main day(1:end-1) '_processing' filesep filename 'img3' filesep 'img_' num2str(kkkk) '_' num2str(kkk) '_' num2str(kk) '_' sprintf('%02d',k) '.tif'],options);
                    end
                end
            end
        end
    else
        warning('Tif files were already saved in destination folder, please delete to overwrite.')
    end

    matlab.apputil.run('PIVlab')
end

% click 'Load images', select images to be processed, click 'add', click 'import'
% in the new window, ...

%% PIVlab GUI: analyse data

if GUIflag
    sz=size(img3);
    %check sequence style 1-2,2-3,3-4...

    % edit filename with parameters entered in GUI
    filenamePIV='pivResults_stretchLim_CLAHEyes_001739_015_pass1_32x16_pass2_8x4_full_3stdpostprocessing_GUIbased'; % enter parameters in this file name

    mkdir([main day(1:end-1) '_processing' filesep filename 'img3']);

    u=cell2mat(u_original);
    s=size(u_original{1});
    uu=permute(reshape(u.',s(2),s(1),length(u_original)),[2 1 3]);
    uu(:,:,2:length(u_original)+1)=uu; %lacking one transition at end of PIV routine
    uu=reshape(uu,s(1),s(2),sz(3),sz(4),sz(5),[]);

    v=cell2mat(v_original);
    s=size(v_original{1});
    vv=permute(reshape(v.',s(2),s(1),length(v_original)),[2 1 3]);
    vv(:,:,2:length(u_original)+1)=vv;
    vv=reshape(vv,s(1),s(2),sz(3),sz(4),sz(5),[]);
    save([main day(1:end-1) '_processing' filesep filename 'img3' filesep filenamePIV '.mat'],'uu','vv');
end