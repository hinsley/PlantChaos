%close all 

set(0,'DefaultFigureWindowStyle','normal')
fig1   = openfig('Bif_Melibe_Dec7als.fig','invisible');
fig2   = openfig('carter_overlap_diagram.fig','invisible');
%  Prepare 'subplot'
figure (11)
clf
h1=subplot(1,1,1);
copyobj(allchild(get(fig2,'CurrentAxes')),h1)
hold on
copyobj(allchild(get(fig1,'CurrentAxes')),h1)
hold on

axis([-65 100 -4 0.7])
axis([-60 100 -4 .7])
ax = gca;
ax.FontSize = 14;
xlabel('\Delta [Ca]','Fontsize', 18),
ylabel('\Delta V_x','Fontsize', 18) 
box on
axis on
savefig('carter_hil_overlap.fig')