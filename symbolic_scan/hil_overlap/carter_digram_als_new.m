

clear all;
%close all


% figure('units','points','outerposition',[10 10 800 800]); 
% set(gcf, 'color', 'black');
% hold on 
% clf 
figure(2)
clf
%A = imread('new_lyap_fig.png');
%B = imread('new_lyap_light.png');
C = imread('ISI_and_LZ_plot.png');
image(C,'XData',[-60,100],'YData',[1,-4])
hold on

view(0,-90)


%% homoclinic line
homoclinic = load('Homoclinic.mat', 'x');
homoclinic = homoclinic.x;
homoclinic_xshift = homoclinic(973,:);
homoclinic_Cashift = homoclinic(974,:);
% take only values of Cashift<400
homoclinic_xshift = homoclinic_xshift(homoclinic_Cashift < 400);
homoclinic_Cashift = homoclinic_Cashift(homoclinic_Cashift < 400);
% take every third value (still over 10k)
homoclinic_xshift = homoclinic_xshift(3:3:end);
homoclinic_Cashift = homoclinic_Cashift(3:3:end);
clear('homoclinic');
plot (homoclinic_Cashift,homoclinic_xshift,'Color',[1 .5 .0 ],'LineWidth',2.)
hold on
star=5800;
plot (homoclinic_Cashift(star:end),homoclinic_xshift(star:end),'Color',[0 .0 1 ],'LineWidth',2.)
hold on
star=7500;
plot (homoclinic_Cashift(star:end),homoclinic_xshift(star:end),'Color',[1 .5 0 ],'LineWidth',2.)
hold on

%% andronov-hopf line
hopf = load('Hopf.mat', 'x');
hopf = hopf.x;
hopf_xshift = hopf(7,:);
hopf_Cashift = hopf(8,:);
% take only values of Cashift<400
hopf_xshift = hopf_xshift(hopf_Cashift < 400);
hopf_Cashift = hopf_Cashift(hopf_Cashift < 400);
clear('hopf');
BP=1063;
plot (hopf_Cashift(1:BP),hopf_xshift(1:BP),'-.','Color',[.0 .9 .0 ],'LineWidth',2)
hold on
BP=1063;
plot (hopf_Cashift(BP:end),hopf_xshift(BP:end),'Color',[.0 .9 .0 ],'LineWidth',2)
hold on


% snic line
snic = load('SNIC.mat', 'x');
snic = snic.x;
snic_xshift = snic(7,:);
snic_Cashift = snic(8,:);
% take only values of Cashift<400
snic_xshift = snic_xshift(snic_Cashift < 400);
snic_Cashift = snic_Cashift(snic_Cashift < 400);
clear('snic');
plot (snic_Cashift,snic_xshift,'--','Color',[.9 .1 .99 ],'LineWidth',2)
hold on

%% snpo line
snpo_left = load('SNPO_1.mat', 'x');
snpo_left = snpo_left.x;
snpo_xshift_left = snpo_left(968,:);
snpo_Cashift_left = snpo_left(969,:);
clear('snpo_left');

snpo_right = load('SNPO_2.mat', 'x');
snpo_right = snpo_right.x;
snpo_xshift_right = snpo_right(968,:);
snpo_Cashift_right = snpo_right(969,:);
clear('snpo_right');

snpo_xshift = [snpo_xshift_left snpo_xshift_right];
clear('snpo_xshift_left','snpo_xshift_right');
snpo_Cashift = [snpo_Cashift_left snpo_Cashift_right];
clear('snpo_Cashift_left','snpo_Cashift_right');

Fals=1090;
plot (snpo_Cashift(Fals:end),snpo_xshift(Fals:end),'Color',[.0 .4 .99 ],'LineWidth',3)
hold on

% %% Isochrons
% 
% load('period200K_1.mat')
% xshift=x(end-1,:);
% Cashift=x(end,:);
% plot(Cashift,xshift,'Color',[.9 .0 .0 ],'LineWidth',2)
% 
% load('period270K.mat')
% xshift=x(end-1,:);
% Cashift=x(end,:);
% Bel=30;
% plot(Cashift(1:Bel),xshift(1:Bel),'Color',[.9 .0 .0 ],'LineWidth',2)
% hold on 
% %plot(Cashift(Bel),xshift(Bel),'.','MarkerSize',20,'Color',[.9 .0 .0 ])
% hold on 
% Bau=1295;
% plot(Cashift(Bau:end),xshift(Bau:end),'-.','Color',[.9 .9 .9 ],'LineWidth',1)
% hold on 
% load('period270K.mat')
% xshift=x(end-1,:);
% Cashift=x(end,:);
% Bel=30;
% %plot (Cashift(1:Bel),xshift(1:Bel),'Color',[0 0 0],'LineWidth',4)
% hold on
% st=30;
% plot(Cashift(st:end),xshift(st:end),'--','Color',[.6 .6 .6 ],'LineWidth',1)
% hold on 

% load('iso1spike1.mat')
% xshift1=x(end-1,:);
% Cashift1=x(end,:);
% plot(Cashift1(1:end),xshift1(1:end),'Color',[.9 .1 .0 ],'LineWidth',2)
% hold on 
% plot(Cashift1(1),xshift1(1),'.','Color',[.5 .1 .0 ],'MarkerSize',62)
% hold on
% 
% 
% load('iso1spike2.mat')
% xshift2=x(end-1,:);
% Cashift2=x(end,:);
% plot(Cashift2(1:end),xshift2(1:end),'Color',[.9 .1 .0 ],'LineWidth',2)
% hold on
% plot(Cashift2(1),xshift2(1),'.','Color',[.9 .1 .0 ],'MarkerSize',22)
% hold on


load('iso1spike_T72K_a.mat')
xshift3=x(end-1,:);
Cashift3=x(end,:);
plot(Cashift3(1:end),xshift3(1:end),'Color',[.9 .1 .0 ],'LineWidth',2)
hold on 
load('iso1spike_T72K_b.mat')
xshift4=x(end-1,:);
Cashift4=x(end,:);
plot(Cashift4(1:end),xshift4(1:end),'Color',[.9 .1 .0 ],'LineWidth',2)
hold on 


% load('iso_hom_lc_a.mat')
% xshift1=x(end-1,:);
% Cashift1=x(end,:);
% plot(Cashift1(1:end),xshift1(1:end),'Color',[.9 .9 .0 ],'LineWidth',2)
% hold on 
% load('iso_hom_lc_b.mat')
% xshift2=x(end-1,:);
% Cashift2=x(end,:);
% plot(Cashift2(1:end),xshift2(1:end),'Color',[.9 .9 .0 ],'LineWidth',2)
% hold on 


%% homoclinic line
homoclinic = load('Homoclinic.mat', 'x');
homoclinic = homoclinic.x;
homoclinic_xshift = homoclinic(973,:);
homoclinic_Cashift = homoclinic(974,:);
% take only values of Cashift<400
homoclinic_xshift = homoclinic_xshift(homoclinic_Cashift < 400);
homoclinic_Cashift = homoclinic_Cashift(homoclinic_Cashift < 400);
% take every third value (still over 10k)
homoclinic_xshift = homoclinic_xshift(3:3:end);
homoclinic_Cashift = homoclinic_Cashift(3:3:end);
clear('homoclinic');
plot (homoclinic_Cashift,homoclinic_xshift,'Color',[1 .5 .0 ],'LineWidth',2.)
hold on
star=5800;
plot (homoclinic_Cashift(star:end),homoclinic_xshift(star:end),'Color',[0 .0 1 ],'LineWidth',2.)
hold on
star=7500;
plot (homoclinic_Cashift(star:end),homoclinic_xshift(star:end),'Color',[1 .5 0 ],'LineWidth',2.)
hold on

Parabola=xlsread('Shilnikov-Hopf parabola1.xlsx');

x1= Parabola(:,2);
Ca1=Parabola(:,1);
plot(Ca1,x1,'Linewidth',2,'Color',[.1 .1 .1 ])
hold on

%% Belyakov point - double root

% Bel1=0;
% plot(Cashift(1:Bel),xshift(1:Bel),'Color',[.9 .0 .0 ],'LineWidth',2)
% hold on 
bel_xshift = 0.58;
bel_Cashift = -53.49;
plot(bel_Cashift,bel_xshift,'.','MarkerSize',40,'Color',[.5 .0 .0 ])
hold on
plot(bel_Cashift,bel_xshift,'.','MarkerSize',30,'Color',[.85 .0 .0 ])
hold on

bel_xshift1 = -1.24604;
bel_Cashift1 = -34.316;
plot(bel_Cashift1,bel_xshift1,'.','MarkerSize',40,'Color',[.5 .0 .0 ])
hold on
plot(bel_Cashift1,bel_xshift1,'.','MarkerSize',30,'Color',[.85 .0 .0 ])
hold on

%% Bautin points

gh_xshift = hopf_xshift(1063);
gh_Cashift = hopf_Cashift(1063);
plot(gh_Cashift,gh_xshift,'.','MarkerSize',40,'Color',[.0 .5 .0 ])
hold on
plot(gh_Cashift,gh_xshift,'.','MarkerSize',30,'Color',[.0 .8 .0 ])
hold on

gh_xshift = hopf_xshift(8011);
gh_Cashift = hopf_Cashift(8011);
plot(gh_Cashift,gh_xshift,'.','MarkerSize',30,'Color',[.0 .5 .0 ])
hold on


%% Bogdanov-Takens point

bt_xshift = hopf_xshift(2);
bt_Cashift = hopf_Cashift(2);
plot(bt_Cashift,bt_xshift,'.','MarkerSize',30,'Color',[.0 .0 .5 ])
hold on



%% Plot attributes
xlabel('\Delta_{Ca}','FontSize',20);
ylabel('\Delta V_{x}','FontSize',20);
%xlim([-45 -20])
%ylim([-1.5 -0.5])
%axis([-45 -20 -1.5 -0.5])
axis([-64 100 -4 1])

 ax = gca;
ax.FontSize = 12;

axis on
text(-56.8,-1,'SS','Color',[.99 .1 .99 ],'FontSize',16)
text(-50,-3.0,'AH_{sub}','Color',[.0 .9 .0 ],'FontSize',16)
text(82,-3.47,'AH_{super}','Color',[.0 .85 .0 ],'FontSize',16)
text(25,-2.82,'BP','Color',[.0 .9 .0 ],'FontSize',16)
text(-38,-1.38,'Sh-H','Color',[.95 .0 .0 ],'FontSize',16)
text(-49.9,0.59,'S-SF','Color',[.5 .0 .0 ],'FontSize',16)
text(-37.8,-0.75,'homSF','Color',[.9 .0 .0 ],'FontSize',18)
text(8.5,-3.7,'homS_{sl}','Color',[.9 .5 .0 ],'FontSize',16)
text(55,-2.8,'homSN_{po}','Color',[.0 .4 .99 ],'FontSize',16)

savefig('carter_overlap_diagram.fig')