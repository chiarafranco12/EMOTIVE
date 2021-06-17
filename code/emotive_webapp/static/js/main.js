//User menu
const usermenu = document.querySelector('#user-menu');
const pic = document.querySelector('#pic');
pic.addEventListener('click', () =>{
    if(usermenu.classList.contains('hidden')){
        usermenu.classList.remove('hidden');
    }else{
        usermenu.classList.add('hidden');
    }
})
//Navbar menu
const mobilemenu = document.querySelector('#mobile-menu');
const burger = document.querySelector('#burger');
burger.addEventListener('click', () =>{
    if(mobilemenu.classList.contains('hidden')){
        mobilemenu.classList.remove('hidden');
    }
    else{
        mobilemenu.classList.add('hidden');
    }
})
