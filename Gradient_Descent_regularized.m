function Gradient_Descent_regularized(X,Y)

% calculate w and b
[X_n,X_f]=size(X);

lambda=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5];
step_size=[0.001,0.01,0.05,0.1,0.5];
re_CE=zeros(50,length(step_size));     %regularized corss-entropy error

l2_norm=zeros(1,length(step_size));
T=50;

e=exp(1);

lambda=lambda(3);

% regularized model
for s=1:length(step_size)
    w=zeros(X_f,T+1);
    b=[0.1,zeros(1,T)];
    step=step_size(s);
    for t=1:T
        sum_w=zeros(X_f,1);
        sum_b=0;
        
        for n=1:X_n
            
            sigmoid=1/(1+e^(-(w(:,t)'*X(n,:)'+b(t))));
            
            sum_b=sum_b+sigmoid-Y(n);
            sum_w=sum_w+(sigmoid-Y(n))*X(n,:)';
            
        end
        
        w(:,t+1)=w(:,t)-step*sum_w-2*lambda*step*(w(:,t));
        b(t+1)=b(t)-step*sum_b;
        
        for n=1:X_n
            
            sigmoid=1/(1+e^(-(w(:,t+1)'*X(n,:)'+b(t+1))));
            if sigmoid<10^(-16)
                sigmoid=10^(-16);
            elseif sigmoid>(1-10^(-16))
                sigmoid=1-10^(-16);
            end
            re_CE(t,s)=re_CE(t,s)-Y(n)*log(sigmoid)-(1-Y(n))*log(1-sigmoid);
            
        end
        re_CE(t,s)=re_CE(t,s)+lambda*(norm(w(:,t+1)))^2;
    end
    l2_norm(s)=norm(w(:,51));
end
%plot cross-entropy function value
x=1:50;
plot(x,re_CE(:,1),'-.b',x,re_CE(:,2),'-ro',x,re_CE(:,3),'gx-',x,re_CE(:,4),'m*-',x,re_CE(:,5),'k+-')
title('Regularized Cross-Entropy value for train data');
xlabel('T');
ylabel('Cross-Entropy value');
legend('0.001','0.01','0.05','0.1','0.5');


