function Gradient_Descent(X,Y)

% calculate w and b
[X_n,X_f]=size(X);

un_CE=zeros(50,5);     %unregularized corss-entropy error
step_size=[0.001,0.01,0.05,0.1,0.5];
l2_norm=zeros(1,5);
T=50;

e=exp(1);

% unregularized model
for s=1:length(step_size)
    w=zeros(X_f,T+1);
    b=0.1;
    
    for t=1:T
        sum_w=zeros(X_f,1);
        sum_b=0;
        
        for n=1:X_n
            
            sigmoid=1/(1+e^(-(w(:,t)'*X(n,:)'+b)));
            
            sum_b=sum_b+sigmoid-Y(n);
            sum_w=sum_w+(sigmoid-Y(n))*X(n,:)';
            
        end
        
        w(:,t+1)=w(:,t)-step_size(s)*sum_w;
        b=b-step_size(s)*sum_b;
        
        for n=1:X_n
            
            sigmoid=1/(1+e^(-(w(:,t+1)'*X(n,:)'+b)));
            if sigmoid<10^(-16)
                sigmoid=10^(-16);
            elseif sigmoid>(1-10^(-16))
                sigmoid=1-10^(-16);
            end
            un_CE(t,s)=un_CE(t,s)-Y(n)*log(sigmoid)-(1-Y(n))*log(1-sigmoid);
            
        end
    end
    l2_norm(s)=norm(w(:,51));
end
% plot cross-entropy function value
x=1:50;
plot(x,un_CE(:,1),'-.b',x,un_CE(:,2),'-ro',x,un_CE(:,3),'gx-',x,un_CE(:,4),'m*-',x,un_CE(:,5),'k+-')

title('Unregularized Cross-Entropy value for train data ');
xlabel('T');
ylabel('Cross-Entropy value');
legend('0.001','0.01','0.05','0.1','0.5');

l2_norm

% regularized model