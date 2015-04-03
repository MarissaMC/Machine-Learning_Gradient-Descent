function Gradient_Descent_regularized_L2(X,Y)

% calculate w and b
[X_n,X_f]=size(X);

lambda=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5];
step_size=0.01;
re_CE=zeros(50,length(lambda));     %regularized corss-entropy error


l2_norm=zeros(1,length(lambda));
T=50;

e=exp(1);

% regularized model
for l=1:length(lambda)
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
        
        w(:,t+1)=w(:,t)-step_size*sum_w-2*lambda(l)*step_size*w(:,t);
        b=b-step_size*sum_b;
        
        for n=1:X_n
            
            sigmoid=1/(1+e^(-(w(:,t+1)'*X(n,:)'+b)));
            if sigmoid<10^(-16)
                sigmoid=10^(-16);
            elseif sigmoid>(1-10^(-16))
                sigmoid=1-10^(-16);
            end
            re_CE(t,l)=re_CE(t,l)-Y(n)*log(sigmoid)-(1-Y(n))*log(1-sigmoid);
            
        end
        re_CE(t,l)=re_CE(t,l)+lambda(l)*norm(w(:,t+1))^2;
    end
    
    l2_norm(l)=norm(w(:,51));
end

l2_norm